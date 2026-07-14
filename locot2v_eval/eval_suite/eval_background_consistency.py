import os
import sys
import json
import argparse
import numpy as np
import cv2
import traceback
from tqdm import tqdm
from PIL import Image

ROOT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_ABS_PATH)
from locot2v_eval.modules.fg_clip import (
    FgCLIP2
)


def get_background_consistency(video_path: str, clip: FgCLIP2):
    """
    Given that our evaluation videos are much longer than that used in VBench, 
    we employ stream-based vide reading to avoid OOM errors. 
    """  
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    prev_img = None
    total_score = 0.0
    pair_cnt = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR->RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        curr_img = Image.fromarray(frame)
        
        if prev_img is not None:
            sim = clip.get_image_similarity(prev_img, curr_img)
            total_score += sim
            pair_cnt += 1

        prev_img = curr_img

    cap.release()

    if pair_cnt == 0:
        raise RuntimeError("Video has less than 2 frames.")

    return total_score / pair_cnt


def get_args():
    parser = argparse.ArgumentParser(description="Background consistency evaluation launched.")
    parser.add_argument("--model_name_or_path", type=str, default=f"{ROOT_ABS_PATH}/ckpts/fg-clip2-base", help="Path to save the FG-CLIP2.")
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
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    fgclip2 = FgCLIP2(args.model_name_or_path)
    with open(args.eval_data, 'r') as f1, open(args.result_path, 'w') as f2:
        test_samples = json.load(f1)
        for test_id, video_path in tqdm(test_samples.items()):
            try:
                results["sample_scores"][test_id] = get_background_consistency(video_path, fgclip2)
                total_score += results["sample_scores"][test_id]
            except Exception as e:
                print(f"{str(e)} encountered when processing {video_path}! Ignoring this video...")
                traceback.print_exc()
        total_score = round(total_score / len(test_samples), 4)
        results["overall_score"] = total_score
        json.dump(results, f2, indent=4, ensure_ascii=False)
        print(f"Average background consistency is: {total_score:.4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)