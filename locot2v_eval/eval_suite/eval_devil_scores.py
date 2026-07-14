import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import subprocess
import tempfile

from typing import List
from tqdm import tqdm
from functools import partial

ROOT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_ABS_PATH)

from locot2v_eval.modules.devil import (
    get_video_data_and_length,
    calc_acf,
    cal_segment,
    get_segments,
    to_same_size
)
from locot2v_eval.modules.viclip import (
    get_viclip, 
    get_vid_feat
)
from locot2v_eval.utils.video_process import (
    pyscenedetect_split_videos_adaptive,
    min_frame_filter
)
from locot2v_eval.utils.utils import (
    safe_rmtree,
    normalize_negative_exponential,
    normalize_power
)


DINOV2_REPO_DIR = "./locot2v_eval/utils/third_party/dinov2"

class DynamicsScorer:
    def __init__(
        self,
        dino_v2_path: str,
        viclip_path: str,
        device="cuda"
    ):
        self.device = device   
        # Load all pretrained models
        self.load_models(
            dino_v2_path=dino_v2_path,
            viclip_path=viclip_path
        )

    def load_models(
        self,
        dino_v2_path: str,
        viclip_path: str,
    ):
        print("Loading Dynamics Models (this may take a while)...")
        self.dino_v2 = self.build_dinov2(dino_v2_path)
        self.viclip = self.build_viclip(viclip_path)

    def get_dynamics_score(self, video_path: str, save_dir: str):
        score_dict = {
            "segment_score": 0.0,
            "video_score": 0.0,
            "segment_dino_score": 0.0,
            "segment_viclip_score": 0.0,
            "video_info_variance_score": 0.0,
            "video_temporal_entropy_score": 0.0
        }
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        scene_list = pyscenedetect_split_videos_adaptive(video_path=video_path, segment_dir=save_dir, filter_func=min_frame_filter)
        
        def get_avg_score(score_list: List[float]):
            if not score_list:
                return 0.0
            return sum(score_list) / len(score_list)
        
        if len(scene_list) > 0:
            segment_paths = []
            for i in range(1, len(scene_list) + 1):
                segment_path = os.path.join(save_dir, f"{video_name}-Scene-{i:03d}.mp4")
                segment_paths.append(segment_path)
            video_data, video_lengths = get_video_data_and_length(segment_paths)
            if video_data is not None:
                video_data = video_data.to(self.device)
            dino_segment_scores = self.cal_dino_segment(video_data, video_lengths)
            viclip_segment_scores = self.cal_viclip_segment(video_data, video_lengths)
            info_variance_scores = self.cal_info_variance(video_data, video_lengths)
            temporal_entropy_dir = os.path.join(save_dir, "temporal_entropy")
            os.makedirs(temporal_entropy_dir, exist_ok=True)
            temporal_entropy_scores = self.cal_temporal_entropy(segment_paths, output_folder=temporal_entropy_dir)
            # segment-level score
            segment_dino_score = get_avg_score(dino_segment_scores)
            segment_dino_score = normalize_power(segment_dino_score, power_val=0.3)
            segment_viclip_score = get_avg_score(viclip_segment_scores)
            segment_viclip_score = normalize_power(segment_viclip_score, power_val=0.3)
            score_dict["segment_dino_score"] = segment_dino_score
            score_dict["segment_viclip_score"] = segment_viclip_score
            score_dict["segment_score"] = (score_dict["segment_dino_score"] + score_dict["segment_viclip_score"]) / 2
            # video-level score
            video_info_variance_score = get_avg_score(info_variance_scores)
            video_info_variance_score = normalize_power(video_info_variance_score, power_val=0.3)
            video_temporal_entropy_score = get_avg_score(temporal_entropy_scores)
            video_temporal_entropy_score = normalize_negative_exponential(video_temporal_entropy_score, scale_k=40000)
            score_dict["video_info_variance_score"] = video_info_variance_score
            score_dict["video_temporal_entropy_score"] = video_temporal_entropy_score
            score_dict["video_score"] = (score_dict["video_info_variance_score"] + score_dict["video_temporal_entropy_score"]) / 2
            
        return score_dict

    def cal_dino_segment(self, video_data: torch.Tensor | None, video_lengths: List[int], batch_size: int=32, min_frame_latency: int=8):
        """
        return List[str]
        """
        all_features = []
        with torch.no_grad():
            for i in range(0, video_data.shape[0], batch_size):
                batch = video_data[i : i + batch_size].to(video_data.device)
                batch_features = self.dino_v2.get_intermediate_layers(batch, n=1)[0]
                all_features.append(batch_features.cpu())
                del batch
                
        features = torch.cat(all_features, dim=0)
        
        features = F.normalize(features, dim=-1, p=2)
        features = features.split(video_lengths)
        distances = []
        for feat in features:
            if feat.shape[0] >= min_frame_latency:
                acf = [calc_acf(feat, k) for k in range(feat.shape[0] // min_frame_latency, feat.shape[0])]
            else:
                acf = []
            acf = sum(acf) / len(acf) if acf else 0
            distances.append(acf)
        
        return distances

    def cal_viclip_segment(self, video_data: torch.Tensor | None, video_lengths: List[int], block_lengths_ratio: List[float]=[0.25], min_segment_length=8):
        """
        return List[float]
        """
        distances = []
        for video in video_data.split(video_lengths):
            if video.shape[0] < min_segment_length:
                print(f"Warning: Video too short ({video.shape[0]} frames), skipping ViClip score.")
                continue

            dist_obj = 0
            for r in block_lengths_ratio:
                segments = get_segments(video, r)
                segments = to_same_size(segments)
                try:
                    seg_feats = get_vid_feat(segments, self.viclip)
                except BaseException as e:
                    print(f"Error extracting ViClip features: {e}")
                    continue
                dist_obj += cal_segment(seg_feats).item()
            distances.append(1 - dist_obj / len(block_lengths_ratio))
        
        return distances

    def cal_info_variance(self, video_data: torch.Tensor | None, video_lengths: List[int], batch_size: int=32):
        """
        return List[float]
        """
        all_features = []
        with torch.no_grad():
            for i in range(0, video_data.shape[0], batch_size):
                batch = video_data[i : i + batch_size].to(video_data.device)
                batch_features = self.dino_v2(batch)
                all_features.append(batch_features.cpu())
                del batch
        features = torch.cat(all_features, dim=0)

        features = F.normalize(features, dim=-1, p=2)
        features = features.split(video_lengths)
        features_mean = [feat.mean(axis=0, keepdim=True) for feat in features]
        distances = [(1 - F.cosine_similarity(feat_mean[:, None], feat[None], dim=-1)).mean(1).item() 
                    for feat, feat_mean in zip(features, features_mean)]
        
        return distances
    
    def cal_temporal_entropy(self, video_paths: List[str], output_folder: str, bash_path: str="./locot2v_eval/utils/third_party/DEVIL/cal_temporal_info.sh"):
        """
        return List[float]
        """
        os.makedirs(output_folder, exist_ok=True)
        results = []
        for video_path in video_paths:
            args = [video_path, output_folder]
            result = subprocess.run(['bash', bash_path] + args, stdout=subprocess.PIPE, text=True)
            try:
                results.append(float(result.stdout.strip()))
            except:
                results.append(0.0)

            safe_rmtree(output_folder)
        
        return results
                    
    def build_dinov2(self, model_name_or_path: str):
        model = torch.hub.load(DINOV2_REPO_DIR, 'dinov2_vitl14', source='local', pretrained=False)
        state_dict = torch.load(model_name_or_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model
    
    def build_viclip(self, model_name_or_path: str):
        cfg = {
            'size': 'l',
            'pretrained': model_name_or_path,
        }
        viclip = get_viclip(cfg['size'], cfg['pretrained'])['viclip']
        viclip = viclip.to(self.device)
        viclip.eval()
        return viclip
        

def get_args():
    parser = argparse.ArgumentParser(description="Devil score evaluation launched.")
    parser.add_argument('--eval_data', type=str, required=True, help='Directory containing video files.')
    parser.add_argument("--temp_dir", type=str, default=f"{ROOT_ABS_PATH}/temp/devil_scores", help="Temporary directory to save the intermediate splits genearted by pyscenedetect.")
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--viclip_path', type=str, default=f'{ROOT_ABS_PATH}/ckpts/devil_weights/ViClip-InternVid-10M-FLT.pth')
    parser.add_argument('--dino_v2_path', type=str, default=f'{ROOT_ABS_PATH}/ckpts/devil_weights/dinov2_vitl14_pretrain.pth')
    
    args = parser.parse_args()
    return args


def main(args):
    assert os.path.exists(args.eval_data), f"Failed to find corresponding file of {args.eval_data}, please carefully check your path again!"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {
        "overall_segment_score": 0.0,
        "overall_video_score": 0.0,
        "overall_segment_dino_score": 0.0,
        "overall_segment_viclip_score": 0.0,
        "overall_video_info_variance_score": 0.0,
        "overall_video_temporal_entropy_score": 0.0,
        "sample_scores": {}
    }
    eval_name = os.path.splitext(os.path.basename(args.eval_data))[0]
    temp_dir = os.path.join(args.temp_dir, eval_name)
    os.makedirs(temp_dir, exist_ok=True)
    
    total_segment_score = 0.0
    total_video_score = 0.0
    total_segment_dino_score = 0.0
    total_segment_viclip_score = 0.0
    total_video_info_variance_score = 0.0
    total_video_temporal_entropy_score = 0.0
    
    # initialize the Scorer
    scorer = DynamicsScorer(
        dino_v2_path=args.dino_v2_path,
        viclip_path=args.viclip_path,
        device=device
    )
    
    with open(args.eval_data, 'r') as f1, open(args.result_path, 'w') as f2:
        test_samples = json.load(f1)
        for test_id, video_path in tqdm(test_samples.items()):
            score_dict = {
                "segment_score": 0.0,
                "video_score": 0.0,
                "segment_dino_score": 0.0,
                "segment_viclip_score": 0.0,
                "video_info_variance_score": 0.0,
                "video_temporal_entropy_score": 0.0
            }
            with tempfile.TemporaryDirectory(dir=temp_dir) as tmp_dir:
                score_dict = scorer.get_dynamics_score(video_path=video_path, save_dir=tmp_dir)
                total_segment_dino_score += score_dict["segment_dino_score"]
                total_segment_viclip_score += score_dict["segment_viclip_score"]
                total_video_info_variance_score += score_dict["video_info_variance_score"]
                total_video_temporal_entropy_score += score_dict["video_temporal_entropy_score"]
                
                total_segment_score += score_dict["segment_score"]
                total_video_score += score_dict["video_score"]
                
                results["sample_scores"][test_id] = score_dict

        results["overall_segment_dino_score"] = round(total_segment_dino_score / len(test_samples), 4)
        results["overall_segment_viclip_score"] = round(total_segment_viclip_score / len(test_samples), 4)
        results["overall_video_info_variance_score"] = round(total_video_info_variance_score / len(test_samples), 4)
        results["overall_video_temporal_entropy_score"] = round(total_video_temporal_entropy_score / len(test_samples), 4)
        
        results["overall_segment_score"] = round(total_segment_score / len(test_samples), 4)
        results["overall_video_score"] = round(total_video_score / len(test_samples), 4)

        json.dump(results, f2, indent=4)

        print(f"Average Segment-level Dynamic Score: {results['overall_segment_score']:.4f}")
        print(f"Average Video-level Dynamic Score: {results['overall_video_score']:.4f}")
        
    safe_rmtree(temp_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)