import argparse
import os
import cv2
import glob
import numpy as np
import torch
import json
import traceback
from tqdm import tqdm
from easydict import EasyDict as edict

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder

ROOT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DynamicDegree:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.load_model()
    
    def load_model(self):
        self.model = RAFT(self.args)
        ckpt = torch.load(self.args.model, map_location="cpu")
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(self.device)
        self.model.eval()

    def get_score(self, img, flo):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()

        u = flo[:,:,0]
        v = flo[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h*w*0.05)

        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])

        return max_rad.item()

    def set_params(self, frame, count):
        scale = min(list(frame.shape)[-2:])
        self.params = {"thres":6.0*(scale/256.0), "count_num":round(4*(count/16.0))}

    def infer_streaming(self, video_path):
        """
        Given that our evaluation videos are much longer than that used in VBench, 
        we employ stream-based vide reading to avoid OOM errors. 
        """
        assert video_path.endswith('mp4'), "Input video must be mp4 format!"
        with torch.no_grad():
            # 1. Open the video and get basic information
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            interval = max(1, round(fps / 8))
            
            # 2. Estimate the number of sampled frames
            total_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            estimated_sampled = max(1, total_video_frames // interval)
            
            # 3. Read the first frame and get the resolution
            ret, first_frame = video.read()
            if not ret:
                raise ValueError(f"Cannot read video: {video_path}")

            # 4. Convert first frame to tensor
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            first_tensor = torch.from_numpy(first_frame_rgb.astype(np.uint8)).permute(2, 0, 1).float()
            first_tensor = first_tensor[None].to(self.device)

            self.set_params(frame=first_tensor, count=estimated_sampled)
            thres = self.params["thres"]
            
            # 5. Reset to the first frame of video and initialize the sum variants
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            total_score_sum = 0.0
            total_pair_count = 0
            prev_tensor = None
            frame_idx = 0
            
            # 6. Read and process the video frame-by-frame
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                if frame_idx % interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    curr_tensor = torch.from_numpy(frame_rgb.astype(np.uint8))
                    curr_tensor = curr_tensor.permute(2, 0, 1).float()
                    curr_tensor = curr_tensor[None].to(self.device)
                    if prev_tensor is not None:
                        padder = InputPadder(prev_tensor.shape)
                        img1, img2 = padder.pad(prev_tensor, curr_tensor)
                        
                        _, flow_up = self.model(img1, img2, iters=20, test_mode=True)
                        max_rad = self.get_score(img1, flow_up)
                        
                        total_score_sum += min(max_rad / thres, 1.0)
                        total_pair_count += 1
                        
                        del flow_up, img1, img2, prev_tensor
                    prev_tensor = curr_tensor
                
                frame_idx += 1
                # periodically clear the gpu cache
                if total_pair_count % 50 == 0 and total_pair_count > 0:
                    torch.cuda.empty_cache()
            
            video.release()
            if prev_tensor is not None:
                del prev_tensor
            torch.cuda.empty_cache()
            
            return total_score_sum / total_pair_count if total_pair_count > 0 else 0.0     

    def check_move(self, score_list):
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list:
            if score > thres:
                count += 1
            if count >= count_num:
                return True
        return False

    def measure_move(self, score_list):
        thres = self.params["thres"]
        move_degree = 0.0
        for score in score_list:
            move_degree += min(score / thres, 1.0)
        return move_degree / len(score_list)


def dynamic_degree(dynamic, video_list):
    sim = []
    video_results = []
    for video_path in tqdm(video_list):
        score_per_video = dynamic.infer_streaming(video_path)
        video_results.append({'video_path': video_path, 'video_results': score_per_video})
        sim.append(score_per_video)
    avg_score = np.mean(sim)
    return avg_score, video_results


def get_args():
    parser = argparse.ArgumentParser(description="Dynamic degree evaluation launched.")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the list file of videos to be evaluated.")
    parser.add_argument("--raft_path", type=str, default=f"{ROOT_ABS_PATH}/ckpts/RAFT/raft-chairs.pth", help="Path to the RAFT model.")
    parser.add_argument("--result_path", type=str, default=f"{ROOT_ABS_PATH}/results/dynamic_degree_results.json", help="Path to save the evaluation results.")
    args = parser.parse_args()
    return args
     
                
def main(args):
    assert os.path.exists(args.eval_data), f"Failed to find corresponding file of {args.eval_data}, please carefully check your path again!"
    results = {
        "overall_score": 0.0,
        "sample_scores": {}
    }
    video_list = []
    args_new = edict({"model": args.raft_path, "small": False, "mixed_precision": False, "alternate_corr":False})
    dynamic = DynamicDegree(args_new, device)
    path_dict = {}
    with open(args.eval_data, 'r') as f:
        test_samples = json.load(f)
        for test_id, video_path in test_samples.items():
            video_list.append(video_path)
            path_dict[video_path] = test_id
        try:
            all_results, video_results = dynamic_degree(dynamic, video_list)
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
    print(f"Average dynamic degree score is: {results['overall_score']: .4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)