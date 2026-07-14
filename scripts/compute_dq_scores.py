#!/usr/bin/env python3
"""
Compute DQ (Dynamic Quality) scores for video samples.

This script takes a directory containing evaluation scores as input and computes
the DQ score for each sample using a pre-trained linear model.

Input files required in the directory:
- eval_motion_smoothness.json
- eval_dynamic_degree.json
- eval_devil_scores.json

Usage:
    python compute_dq_scores.py <directory_path>
    python compute_dq_scores.py results/locot2v_eval/seedance1_5_pro
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Model paths (relative to this script's location)
# SCRIPT_DIR = Path(__file__).parent.absolute()
LINEAR_MODEL = "./ckpts/dq_linear_regression_models/dq_linear_model.pkl"
SCALER_MODEL = "./ckpts/dq_linear_regression_models/dq_scaler.pkl"


def inference_dq_score(input_data):
    """
    Compute DQ score from 6 input metrics using the trained linear model.
    
    input_data: Dictionary format, containing 6 indicators
    Example: {
        "motion_smoothness": 0.8,
        "dynamic_degree": 0.5,
        "segment_dino_score": 0.3,
        "segment_viclip_score": 0.4,
        "video_info_variance_score": 0.1,
        "video_temporal_entropy_score": 0.6
    }
    
    Returns:
        float: Predicted DQ score
    """
    # 1. Load the model and scaler
    model = joblib.load(LINEAR_MODEL)
    scaler = joblib.load(SCALER_MODEL)
    
    # 2. Prepare the data (ensure that the column order is exactly the same as the order of X during training)
    feature_columns = [
        "motion_smoothness", "dynamic_degree", "segment_dino_score", 
        "segment_viclip_score", "video_info_variance_score", "video_temporal_entropy_score"
    ]
    
    # Convert to DataFrame (ensure column names are consistent)
    df_new = pd.DataFrame([input_data])[feature_columns]
    
    # 3. Data normalization (use the scaler from training for transform, do not use fit_transform)
    X_new_scaled = scaler.transform(df_new)
    
    # 4. Obtain the final scores
    predicted_score = model.predict(X_new_scaled)
    
    return predicted_score[0]


def load_json_file(filepath):
    """Load and parse a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_dq_scores(directory_path):
    """
    Compute DQ scores for all samples in the given directory.
    
    Args:
        directory_path: Path to directory containing evaluation score files
        
    Returns:
        dict: Results containing overall DQ score and per-sample DQ scores
    """
    directory_path = Path(directory_path)
    
    # Define input files
    motion_smoothness_file = directory_path / "eval_motion_smoothness.json"
    dynamic_degree_file = directory_path / "eval_dynamic_degree.json"
    devil_scores_file = directory_path / "eval_devil_scores.json"
    
    # Validate input files exist
    missing_files = []
    if not motion_smoothness_file.exists():
        missing_files.append(str(motion_smoothness_file))
    if not dynamic_degree_file.exists():
        missing_files.append(str(dynamic_degree_file))
    if not devil_scores_file.exists():
        missing_files.append(str(devil_scores_file))
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files:\n" + "\n".join(f"  - {f}" for f in missing_files)
        )
    
    # Load all score files
    print(f"Loading scores from: {directory_path}")
    
    motion_smoothness_data = load_json_file(motion_smoothness_file)
    dynamic_degree_data = load_json_file(dynamic_degree_file)
    devil_scores_data = load_json_file(devil_scores_file)
    
    # Get sample names from motion_smoothness (or any of the files)
    sample_names = set(motion_smoothness_data.get("sample_scores", {}).keys())
    sample_names &= set(dynamic_degree_data.get("sample_scores", {}).keys())
    sample_names &= set(devil_scores_data.get("sample_scores", {}).keys())
    
    if not sample_names:
        raise ValueError("No common sample names found across all score files")
    
    print(f"Found {len(sample_names)} samples to process")
    
    # Compute DQ score for each sample
    sample_dq_scores = {}
    for sample_name in sorted(sample_names):
        # Extract the 6 metrics needed for DQ prediction
        input_data = {
            "motion_smoothness": motion_smoothness_data["sample_scores"][sample_name],
            "dynamic_degree": dynamic_degree_data["sample_scores"][sample_name],
            "segment_dino_score": devil_scores_data["sample_scores"][sample_name]["segment_dino_score"],
            "segment_viclip_score": devil_scores_data["sample_scores"][sample_name]["segment_viclip_score"],
            "video_info_variance_score": devil_scores_data["sample_scores"][sample_name]["video_info_variance_score"],
            "video_temporal_entropy_score": devil_scores_data["sample_scores"][sample_name]["video_temporal_entropy_score"]
        }
        
        # Compute DQ score
        dq_score = inference_dq_score(input_data)
        sample_dq_scores[sample_name] = dq_score
    
    # Compute overall DQ score (mean of all sample scores)
    overall_dq_score = float(np.mean(list(sample_dq_scores.values())))
    
    # Prepare results
    results = {
        "overall_dq_score": overall_dq_score,
        "sample_dq_scores": sample_dq_scores
    }
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_dq_scores.py <directory_path>")
        print("Example: python compute_dq_scores.py results/locot2v_eval/seedance1_5_pro")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    try:
        results = compute_dq_scores(directory_path)
        
        # Print results
        print("\n" + "=" * 60)
        print("DQ Score Results")
        print("=" * 60)
        print(f"\nOverall DQ Score: {results['overall_dq_score']:.4f}")
        print("\nPer-sample DQ Scores:")
        for sample_name, score in results['sample_dq_scores'].items():
            print(f"  {sample_name}: {score:.4f}")
        
        # Optionally save results to JSON file
        output_path = Path(directory_path) / "eval_dq_scores.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
