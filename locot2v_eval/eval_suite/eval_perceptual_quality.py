import os
import sys
import json
import torch
import cv2
import numpy as np
import requests
import argparse
import tempfile
from typing import List, Tuple
from transformers import AutoModelForCausalLM
from PIL import Image
from tqdm import tqdm

ROOT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_ABS_PATH)

from locot2v_eval.modules.deqa_score import load_deqa_model, get_deqa_score


UPPER_BOUND = 5.0

def get_sampling_plan(total_frames: int, scales: List[Tuple[float, float]] | None=None):
    """
    Generate dynamic sampling plan.
    :param total_frames: Total number of frames in the video.
    :param scales: List, each element is (window length proportion, sampling proportion within the window).
                   For example, (0.2, 0.05) indicates that the window length is 20% of the total length, and the frame number in the middle 5% of the window is taken.
    """
    # Default configuration: (window ratio, sampling ratio)
    # It is recommended that the sampling ratio should not be too high, otherwise the computational load for long videos will be very large
    if scales is None:
        scales = [
            (0.2, 0.05),  # Macro window: length and width, moderate sampling density
            (0.1, 0.10),  # Medium window: medium width, high sampling density
            (0.05, 0.20)  # Microscopic window: short and wide, with extremely high sampling density (focusing on local areas)
        ]
        
    plan = dict()
    
    for scale_id, (win_ratio, sample_ratio) in enumerate(scales):
        win_len = int(total_frames * win_ratio)
        if win_len < 1:
            continue
            
        # Dynamically calculate the number of samples
        num_samples = int(win_len * sample_ratio)
        # At least one frame is captured from each window
        num_samples = max(1, num_samples)
        
        # Set the step length to the window length to ensure non-overlapping windows (alternatively, you can change it to win_len // 2 for overlapping sampling)
        stride = win_len
        
        for win_id, start in enumerate(range(0, total_frames, stride)):
            end = min(start + win_len, total_frames)
            
            # Calculate the center of the window
            center = (start + end) // 2
            
            # Calculate the start point of the sampling interval to ensure it centers around the midpoint
            sample_start = center - num_samples // 2
            
            # Generate indices and clip them to the boundaries
            # The logic here is to continuously sample a central segment.
            # If you want the central segment to be sparser, you can add a step parameter to arange
            indices = np.arange(sample_start, sample_start + num_samples)
            indices = np.clip(indices, start, end - 1)
            
            # Save into the plan
            for idx in indices:
                idx = int(idx) # Ensure it is an integer
                if idx not in plan:
                    plan[idx] = []
                plan[idx].append((scale_id, win_id))
                
    return plan


def get_video_score(video_path: str, score_model, need_norm: bool=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to load the video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. Get the sampling plan
    plan = get_sampling_plan(total_frames)
    
    # If the video is too short and the plan is empty, force sampling the middle frame
    if not plan and total_frames > 0:
        mid = total_frames // 2
        plan[mid] = [(0, 0)] # Classify as scale 0, window 0

    # 2. Prepare to read frames
    # Sort all target frame indices to read sequentially and reduce disk seek time
    target_indices = sorted(plan.keys())
    
    window_scores = dict() # Structure: {scale_id: {win_id: [scores...]}}
    
    # 3. Optimized read loop
    current_pos = -1 # Track the current decoder position
    
    for frame_idx in target_indices:
        # Core optimization: seek only when the target frame is not the next frame
        # For consecutive frames (e.g., 10 continuous center frames), only the first one needs a seek; the rest are read directly
        if frame_idx != current_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            break # Video read error or end of stream
            
        # Update the current position (the pointer auto-increments after read)
        current_pos = frame_idx + 1
        
        # Model inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Call your model here
        score = get_deqa_score(score_model, img)
        
        # Dispatch scores to corresponding windows
        for scale_id, win_id in plan[frame_idx]:
            if scale_id not in window_scores:
                window_scores[scale_id] = dict()
            if win_id not in window_scores[scale_id]:
                window_scores[scale_id][win_id] = []
            window_scores[scale_id][win_id].append(score)
            
    cap.release()
    
    # 4. Compute average score per scale (Scale -> Window -> Frame)
    scale_means = []
    
    # Iterate over all possible scales (even if some scales have no samples)
    # Dynamically get the actually existing scale_ids
    existing_scales = sorted(window_scores.keys())
    
    for scale_id in existing_scales:
        window_means = []
        for win_id, scores in window_scores[scale_id].items():
            if scores:
                window_means.append(np.mean(scores))
        
        if window_means:
            scale_means.append(np.mean(window_means))
    
    # 5. Final aggregation
    if not scale_means:
        return 0.0 # or other default value
        
    final_score = np.mean(scale_means)
    
    if need_norm:
        return float(final_score) / UPPER_BOUND
    else:
        return float(final_score)


def get_args():
    parser = argparse.ArgumentParser(description="Perceptual Quality Evaluation Launched...")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model name or path to load the evaluation model.")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the list file of videos to be evaluated.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to save the evaluation results.")
    args = parser.parse_args()
    return args


def main(args):
    assert os.path.exists(args.eval_data), f"Failed to find corresponding file of {args.eval_data}, please carefully check your path again!"
    results = {
        "overall_score": 0.0,
        "sample_scores": {}
    }
    total_score = 0.0
    deqa_model = load_deqa_model(args.model_name_or_path)
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    with open(args.eval_data, 'r') as f1, open(args.result_path, 'w') as f2:
        test_samples = json.load(f1)
        for test_id, video_path in tqdm(test_samples.items()):
            try:
                sample_score = get_video_score(video_path, deqa_model)
                total_score += sample_score
                results["sample_scores"][test_id] = sample_score
            except Exception as e:
                print(f"{str(e)} encountered when processing {video_path}! Ignoring this video...")         
        total_score = round(total_score / len(test_samples), 4)
        results["overall_score"] = total_score
        json.dump(results, f2, indent=4, ensure_ascii=False)
        print(f"Average perceptual quality score is: {total_score: .4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)