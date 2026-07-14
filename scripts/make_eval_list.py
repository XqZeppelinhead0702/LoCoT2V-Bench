import os
import json
import argparse
from pathlib import Path


def make_eval_list(video_dir, output_json):
    video_dir = Path(video_dir)
    videos = sorted(video_dir.glob("*.mp4"))
    data = {v.stem: str(v.resolve()) for v in videos}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} entries to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    make_eval_list(args.video_dir, args.output_json)