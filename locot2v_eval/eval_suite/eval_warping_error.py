import os
import sys
import cv2
import torch
import numpy as np
import argparse
import torch.nn.functional as F
import json
import traceback
from tqdm import tqdm

ROOT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_ABS_PATH)
sys.path.append(f"{ROOT_ABS_PATH}/locot2v_eval/utils/third_party/RAFT")

import locot2v_eval.modules.warp_utils as warp_utils
from locot2v_eval.utils.third_party.RAFT.core.raft import RAFT
from locot2v_eval.utils.third_party.RAFT.core.utils import flow_viz
from locot2v_eval.utils.third_party.RAFT.core.utils.utils import InputPadder


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(args):
    assert os.path.exists(args.raft_path), f"Error occurs from load_model in {os.path.abspath(__file__)}, failed to find {args.raft_path} for raft model!"
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.raft_path))

    model = model.module
    model.to(device)
    model.eval()
    model.args.mixed_precision = False
    return model


def get_video_info(video_path):
    """
        Get total frames of video
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def read_frame_at_index(cap, frame_idx):
    """
        Read the frame at target index
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.array(frame)
    return None


def compute_warping_error_streaming(video_path, model):
    """
        Use streaming procedure to reduce the memory usage and avoid OOM error.
        Our code is adapted from [EvalCrafter](https://github.com/evalcrafter/EvalCrafter/tree/master). 
        However, the source code is directly read all frames and calculate warping error, which had always encountered OOM error during our practice.
    """
    total_frames = get_video_info(video_path)
    if total_frames <= 1:
        return 0.0
    Num = total_frames
    indices = torch.linspace(0, total_frames - 1, Num).long().tolist()
    
    cap = cv2.VideoCapture(video_path)
    warping_error = 0
    err = 0
    
    # save previous frame
    prev_frame = None
    prev_idx = None

    with torch.no_grad():
        for i, frame_idx in enumerate(indices):
            # read current frame
            current_frame = read_frame_at_index(cap, frame_idx)
            
            if current_frame is None:
                continue
            
            # if there's a previous frame, calculate warping error
            if prev_frame is not None:
                frame1 = torch.from_numpy(prev_frame)
                frame2 = torch.from_numpy(current_frame)
            
                # Calculate optical flow using Farneback method
                img1 = frame1.permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                img2 = frame2.permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                
                # Downsample the images by a factor of 2
                img1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear', align_corners=False)
                img2 = F.interpolate(img2, scale_factor=0.5, mode='bilinear', align_corners=False)
                
                padder = InputPadder(img1.shape)
                img1, img2 = padder.pad(img1, img2)
                
                ### compute fw flow
                _, fw_flow = model(img1, img2, iters=20, test_mode=True)
                fw_flow = warp_utils.tensor2img(fw_flow)
                
                ### compute bw flow
                _, bw_flow = model(img2, img1, iters=20, test_mode=True)
                bw_flow = warp_utils.tensor2img(bw_flow)
                
                ### compute occlusion
                fw_occ, warp_img2 = warp_utils.detect_occlusion(bw_flow, fw_flow, img2)
                warp_img2 = torch.tensor(warp_img2).float().to(device)
                fw_occ = torch.tensor(fw_occ).float().to(device)
                
                ### load occlusion mask
                occ_mask = fw_occ
                noc_mask = 1 - occ_mask
                
                diff = (warp_img2 - img1) * noc_mask
                diff_squared = diff ** 2
                
                # Calculate the sum and mean
                N = torch.sum(noc_mask)
                if N == 0:
                    N = diff_squared.numel()
                err += torch.sum(diff_squared) / N
                
                # empty gpu cache
                del img1, img2, fw_flow, bw_flow, warp_img2, fw_occ, noc_mask, diff, diff_squared
                torch.cuda.empty_cache()
            
            # update previous frame
            prev_frame = current_frame
            prev_idx = frame_idx
    
    cap.release()
    
    num_pairs = len(indices) - 1
    if num_pairs > 0:
        warping_error = err / num_pairs
    
    return warping_error if not isinstance(warping_error, torch.Tensor) else warping_error.item()


def get_args():
    parser = argparse.ArgumentParser(description="Warping error evaluation launched.")
    parser.add_argument("--eval_data", type=str, default="./eval_videos/eval_list.json")
    parser.add_argument("--raft_path", type=str, default=f"{ROOT_ABS_PATH}/ckpts/RAFT/raft-things.pth")
    parser.add_argument("--result_path", type=str, default=f"{ROOT_ABS_PATH}/results/warping_error_results.json", help="Path to save the evaluation results.")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--norm_alpha', type=float, default=5.0, help='Value of alpha to norm the original warping error into the range [0, 1]')
    args = parser.parse_args()
    
    return args


def main(args):
    assert os.path.exists(args.eval_data), f"Failed to find corresponding file of {args.eval_data}, please carefully check your path again!"
    results = {
        "overall_score": 0.0,
        "sample_scores": {}
    }
    total_score = 0.0
    raft_model = load_model(args)
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    with open(args.eval_data, 'r') as f1, open(args.result_path, 'w') as f2:
        test_samples = json.load(f1)
        for test_id, video_path in tqdm(test_samples.items()):
            try:
                sample_score = compute_warping_error_streaming(video_path, raft_model)
                sample_score = np.exp(args.norm_alpha * -(sample_score))
                total_score += sample_score
                results["sample_scores"][test_id] = sample_score
            except Exception as e:
                print(f"{str(e)} encountered when processing {video_path}! Ignoring this video...")         
                traceback.print_exc()
        total_score = round(total_score / len(test_samples), 4)
        results["overall_score"] = total_score
        json.dump(results, f2, indent=4, ensure_ascii=False)
        print(f"Average warping error is: {total_score: .4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)