import os
import cv2
import glob
import torch
import numpy as np
import argparse
import json
import traceback
from tqdm import tqdm
from omegaconf import OmegaConf

from vbench.third_party.amt.utils.utils import (
    img2tensor, tensor2img,
    check_dim_and_resize
    )
from vbench.third_party.amt.utils.build_utils import build_from_cfg
from vbench.third_party.amt.utils.utils import InputPadder

ROOT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
class MotionSmoothness:
    def __init__(self, config, ckpt, device):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = 1
        self.initialization()
        self.load_model()
        
    def load_model(self):
        cfg_path = self.config
        ckpt_path = self.ckpt
        network_cfg = OmegaConf.load(cfg_path).network
        network_name = network_cfg.name
        print(f'Loading [{network_name}] from [{ckpt_path}]...')
        self.model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def initialization(self):
        if self.device == 'cuda':
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
            print("VRAM available: {:.1f} MB".format(self.vram_avail / 1024 ** 2))
        else:
            # Do not resize in cpu mode
            self.anchor_resolution = 8192*8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        self.embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(self.device)
        # self.fp = FrameProcess()
        
    def motion_score(self, video_path):
        """
        Given that our evaluation videos are much longer than that used in VBench, 
        we employ stream-based vide reading to avoid OOM errors. 
        """
        if not video_path.endswith('.mp4'):
            # we only support mp4 format video for unification
            raise NotImplementedError("Only support video input in stream mode")
        
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video {video_path}"
        
        success, prev_frame = cap.read()
        if not success:
            raise RuntimeError("Failed to read video first frame")

        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        
        padder = None
        output_scores = []
        frame_idx = 1
        orig_frame = None
        with torch.no_grad():
            while True:
                success, next_frame = cap.read()
                if not success:
                    break
                if frame_idx % 2 != 0:
                    orig_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
                    frame_idx += 1
                    continue
                next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
                
                # transformed into tensor
                in_0 = img2tensor(prev_frame).to(self.device)
                in_1 = img2tensor(next_frame).to(self.device)
                
                # padding initialization
                if padder is None:
                    padder = InputPadder(in_0.shape)
                
                in_0, in_1 = padder.pad(in_0, in_1)
                # interpolation and get difference
                pred = self.model(in_0, in_1, self.embt, eval=True)['imgt_pred']
                pred = padder.unpad(pred)
                pred_img = tensor2img(pred)
                diff = self.get_diff(orig_frame, pred_img)
                output_scores.append(diff)

                # manually free the memory 
                del in_0, in_1, pred
                torch.cuda.empty_cache()
                
                # update previous frame
                prev_frame = next_frame
                frame_idx += 1

        cap.release()
        
        vfi_score = np.mean(output_scores)
        norm = (255.0 - vfi_score) / 255.0
        return norm

    def get_diff(self, img1, img2):
        img = cv2.absdiff(img1, img2)
        return np.mean(img)


def motion_smoothness(motion, video_list):
    sim = []
    video_results = []
    for video_path in tqdm(video_list):
        score_per_video = motion.motion_score(video_path)
        video_results.append({'video_path': video_path, 'video_results': score_per_video})
        sim.append(score_per_video)
    avg_score = np.mean(sim)
    return avg_score, video_results


def get_args():
    parser = argparse.ArgumentParser(description="Motion Smoothness Evaluation launched.")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the list file of videos to be evaluated.")
    parser.add_argument("--amt_path", type=str, default=f"{ROOT_ABS_PATH}/ckpts/AMT/amt-s.pth", help="Path to the AMT model.")    
    parser.add_argument("--amt_config", type=str, default=f"{ROOT_ABS_PATH}/ckpts/AMT/AMT-S.yaml", help="Path to the AMT config.")    
    parser.add_argument("--result_path", type=str, required=True, help="Path to save the evaluation results.")
    args = parser.parse_args()
    return args


def main(args):
    assert os.path.exists(args.eval_data), f"Failed to find corresponding file of {args.eval_data}, please carefully check your path again!"
    results = {
        "overall_score": 0.0,
        "sample_scores": {}
    }
    video_list = []
    motion = MotionSmoothness(args.amt_config, args.amt_path, device)
    path_dict = {}
    with open(args.eval_data, 'r') as f:
        test_samples = json.load(f)
        for test_id, video_path in test_samples.items():
            video_list.append(video_path)
            path_dict[video_path] = test_id
        try:
            all_results, video_results = motion_smoothness(motion, video_list)
            results['overall_score'] = round(float(all_results), 4)
            for video_result in video_results:
                v_path = video_result['video_path']
                v_res = video_result['video_results']
                results["sample_scores"][path_dict[v_path]] = float(v_res)
        except Exception as e:
            print(f"{str(e)} encountered when processing {video_path}! Ignoring this video...")
            traceback.print_exc()
    os.makedirs(os.path.abspath(os.path.dirname(args.result_path)), exist_ok=True)
    with open(args.result_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Average motion smoothness score is: {results['overall_score']: .4f}")
    

if __name__ == "__main__":
    args = get_args()
    main(args)
