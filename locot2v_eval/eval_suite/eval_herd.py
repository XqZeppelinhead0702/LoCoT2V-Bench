import os
import sys
import json
import argparse
import torch
import re
import traceback

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from typing import List, Union

ROOT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT_ABS_PATH)

from locot2v_eval.modules.llm import MllmBase, Qwen3VLModel
from locot2v_eval.utils.utils import extract_json_dict


CONTROL_SUFFIX = " Only answer \"Yes\" or \"No\"."
device = 'cuda' if torch.cuda.is_available() else 'cpu'

AUDITOR_PROMPT = """You are an expert at objectively describing video content and serving as a forensic observer of visual details.

Analyze the provided video and provide a factual report based on the following aspects:
1. Subject & Character: Details of appearance, facial expressions, and movement naturalness for each subject or character in the video.
2. Setting & Atmosphere: Background details, lighting, color palette, and environmental consistency.
3. Temporal Stability: Detection of AI-generated artifacts, such as flickering, warping, or objects morphing/disappearing.
4. Narrative Logic: The sequence of events and the smoothness of transitions between scenes.

For each aspect, describe exactly what is visible without any subjective praise or interpretation of intent. Focus on identifying both the content and any technical inconsistencies.

You should only respond with the descriptive report for these four aspects without any other irrelevant content.

Your respone should be like the following format:
[Subject & Character]: A woman in a white lab coat is standing in a garden. Her facial expression is static. As she turns, her hair momentarily clips through her shoulder.
[Setting & Atmosphere]: Daylight setting with green foliage in the background. The lighting is bright and consistent.
[Temporal Stability]: The background trees flicker slightly between the 2-second and 4-second mark. The buttons on the lab coat change from three to four during the camera pan.
[Narrative Logic]: The video consists of a single continuous shot. The camera pans from left to right.
"""

EVALUATOR_TEMP = """You are a senior film critic and a professional evaluator of video-text alignment, specializing in assessing how well a video's execution meets specific thematic and technical goals.

Evaluate how well the video aligns with the provided goals. An Auditor's Report is provided to help you understand and evaluate the video. You must consider both the creative expression and the technical execution (noting that AI artifacts or inconsistencies significantly lower the alignment quality).

Evaluate based on the 6 dimensions defined in the Description:
1. Emotional Response: The expected emotions or feelings the video would aim to evoke in viewers.
2. Narrative Flow: The anticipated structure and pacing of the storytelling, including how smoothly events progress.
3. Character Development: Expectations about how characters or key subjects should be portrayed and how their roles or arcs might evolve.
4. Visual Style: The likely visual atmosphere, including color palette, cinematography, composition, and stylistic choices. But do not include any requirement for the resolution such HD, 4K or 8K.
5. Themes Expression: The core ideas, messages, or commentary the video is expected to convey.
6. Overall Impression: The general expected impact, value, or appeal of the hypothetical video, including who might enjoy it.

Scoring Rules (1-5):
- 5 (Exceptional): Perfect alignment. The video fully realizes the description with professional-level execution and zero or negligible AI artifacts.
- 4 (Strong): High alignment. The intent is clearly achieved, though there are minor technical flaws or slight deviations from the description.
- 3 (Moderate): Partial alignment. The core ideas are present, but the experience is hindered by noticeable AI distortions, stiff movements, or inconsistent details.
- 2 (Weak): Poor alignment. Significant gaps exist between the description and the visuals; the execution is amateurish or logically flawed.
- 1 (Failed): No alignment. The video fails to convey the intended themes or is technically unwatchable.

You should only respond with an integer ranging from 1 to 5 as your scoring result without any other irrelevant content.

Your response could be like the following format:
```json
{{
    "Emotional Response": 2,
    "Narrative Flow": 3,
    "Character Development": 1,
    "Visual Style": 4,
    "Themes Expression": 3,
    "Overall Impression": 2
}}
```

Now here is the Auditor's Report:
{auditor_report}

And here is the provided goals in JSON format:
```json
{herd_expectations}
```
"""
    

class VisualAgent:
    def __init__(self, mllm: MllmBase):
        self.mllm = mllm
    
    def get_response(
        self,
        user_prompt: str, 
        system_prompt: str=None, 
        image_path: Union[str, List[str]]=None, 
        video_path: Union[str, List[str]]=None,
        max_new_tokens: int=4096
    ):
        return self.mllm.get_response(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image_path=image_path,
            video_path=video_path,
            max_new_tokens=max_new_tokens
        )


class HERDScorer:
    def __init__(self, mllm_name_or_path: str):
        mllm = Qwen3VLModel(model_name_or_path=mllm_name_or_path)
        self.auditor_agent = VisualAgent(mllm=mllm)
        self.evaluator_agent = VisualAgent(mllm=mllm)    
    
    def get_herd_score(self, video_path: str, herd_expectations: List[dict]):
        score_dict = {"sample_score": 0.0}
        auditor_res = self.auditor_agent.get_response(user_prompt=AUDITOR_PROMPT, video_path=video_path)
        evaluator_prompt = EVALUATOR_TEMP.format(auditor_report=auditor_res, herd_expectations=json.dumps(herd_expectations, indent=4, ensure_ascii=False))
        evaluator_res = self.evaluator_agent.get_response(user_prompt=evaluator_prompt, video_path=video_path)
        try:
            json_scores = json.loads(evaluator_res)
        except Exception as e:
            json_scores = extract_json_dict(evaluator_res)
        if len(json_scores) == 0:
            print(f"Fail to extract target scores for the response of EvaluatorAgent when processing {video_path}!")
            json_scores = {
                "Emotional Response": 0.0,
                "Narrative Flow": 0.0,
                "Character Development": 0.0,
                "Visual Style": 0.0,
                "Themes Expression": 0.0,
                "Overall Impression": 0.0
            }

        for herd_dim in json_scores.keys():
            json_scores[herd_dim] /= 5
        
        score_dict["sample_score"] = sum(list(json_scores.values())) / len(json_scores)
        score_dict.update(json_scores)
        return score_dict
        
        
def get_args():
    parser = argparse.ArgumentParser(description="Overall video2text alignment evaluation launched.")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the list file of videos to be evaluated.")
    parser.add_argument("--mllm_path", type=str, default=f"{ROOT_ABS_PATH}/ckpts/Qwen3-VL-8B-Instruct", help="Path to save the MLLM used for video scoring task.")
    parser.add_argument("--src_path", type=str, default=f"{ROOT_ABS_PATH}/data/prompt.json", help="Path to source the prompt metadata.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to save the evaluation results.")
    args = parser.parse_args()
    return args


def main(args):
    assert os.path.exists(args.eval_data), f"Failed to find corresponding file of {args.eval_data}, please carefully check your path again!"
    results = {
        "overall_score": 0.0,
        "overall_er_score": 0.0,
        "overall_nf_score": 0.0,
        "overall_cd_score": 0.0,
        "overall_vs_score": 0.0,
        "overall_te_score": 0.0,
        "overall_oi_score": 0.0,
        "sample_scores": {}
    }
    total_score = 0.0
    total_er_score = 0.0
    total_nf_score = 0.0
    total_cd_score = 0.0
    total_vs_score = 0.0
    total_te_score = 0.0
    total_oi_score = 0.0
    herd_scorer = HERDScorer(mllm_name_or_path=args.mllm_path)
    with open(args.eval_data, 'r') as f1, open(args.result_path, 'w') as f2, open(args.src_path, 'r') as f3:
        test_samples = json.load(f1)
        prompt_data = json.load(f3)
        for test_id, video_path in tqdm(test_samples.items()):
            # For test data
            # description_text = prompt_data[test_id]
            theme_name = "_".join(test_id.split('_')[:-1])
            herd_expectations = prompt_data[theme_name][test_id]["herd_expectations"]
            try:
                score_dict = herd_scorer.get_herd_score(video_path, herd_expectations)
                results["sample_scores"][test_id] = score_dict
                total_score += score_dict["sample_score"]
                total_er_score += score_dict["Emotional Response"]
                total_nf_score += score_dict["Narrative Flow"]
                total_cd_score += score_dict["Character Development"]
                total_vs_score += score_dict["Visual Style"]
                total_te_score += score_dict["Themes Expression"]
                total_oi_score += score_dict["Overall Impression"]
            except Exception as e:
                print(f"{str(e)} encountered when processing {video_path}! Ignoring this video...")
                traceback.print_exc()
                results["sample_scores"][test_id] = {}
        total_score = round(total_score / len(test_samples), 4)
        total_er_score = round(total_er_score / len(test_samples), 4)
        total_nf_score = round(total_nf_score / len(test_samples), 4)
        total_cd_score = round(total_cd_score / len(test_samples), 4)
        total_vs_score = round(total_vs_score / len(test_samples), 4)
        total_te_score = round(total_te_score / len(test_samples), 4)
        total_oi_score = round(total_oi_score / len(test_samples), 4)
        
        results["overall_score"] = total_score
        results["overall_er_score"] = total_er_score
        results["overall_nf_score"] = total_nf_score
        results["overall_cd_score"] = total_cd_score
        results["overall_vs_score"] = total_vs_score
        results["overall_te_score"] = total_te_score
        results["overall_oi_score"] = total_oi_score
        
        json.dump(results, f2, indent=4, ensure_ascii=False)   
        print(f"Average HERD score is: {total_score: .4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)