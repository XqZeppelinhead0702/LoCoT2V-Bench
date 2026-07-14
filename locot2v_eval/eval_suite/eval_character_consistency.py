import os
import sys
import json
import heapq
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import imageio
import tempfile
import traceback
import imageio

from typing import Any, List, Dict
from tqdm import tqdm
from transformers import Sam3VideoModel, Sam3VideoProcessor, Qwen3VLForConditionalGeneration, AutoModel, AutoProcessor
from accelerate import Accelerator

ROOT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_ABS_PATH)

from locot2v_eval.modules.fg_clip import FgCLIP2
from locot2v_eval.utils.video_process import load_sampled_frames
from locot2v_eval.utils.utils import safe_rmtree


CONF_THERSHOLD = 0.8

CHARACTER_RECOGNITION_TEMP = "Is the {character_concept} basically matched with the description \"{character_description}\"? Only answer \"Yes\" or \"No\"."

def uniform_sample(lst: List[Any], n):
    if n <= 0:
        return []
    if n == 1:
        return [lst[0]]
    if n >= len(lst):
        return lst.copy()

    L = len(lst)
    step = (L - 1) / (n - 1)

    indices = [round(i * step) for i in range(n)]
    return [lst[i] for i in indices]


def topk_by_conf(lst: List[Dict], n: int):
    if n <= 0:
        return []
    if n >= len(lst):
        return sorted(lst, key=lambda x: x["conf"], reverse=True)
    
    return heapq.nlargest(n, lst, key=lambda x: x["conf"])


class CharConsistencyScorer:
    def __init__(self, sam3_path: str, mllm_path: str, clip_path: str):
        self.device = Accelerator().device
        self.load_sam3(sam3_path)
        self.load_mllm(mllm_path)
        self.load_fgclip2(clip_path)
        
    def compute_character_consistency(self, video_path: str, character_infos: dict, save_dir: str):
        concept_obj_images, character_obj_mapping = self.extract_character_images(video_path, character_infos, save_dir)
        consistency_score = 0
        for v in character_obj_mapping.values():
            character_score = 0
            if len(v) == 0:
                continue
            v_split = v.split('_')
            images = concept_obj_images[v_split[0]][v_split[1]]

            if len(images) == 0:
                # It's impossible that one character has no image, seen as a bad case with zero score
                pass
            elif len(images) < 2:
                # Only have one image? That could be seen as great consistency because no consistency requirements.
                consistency_score += 1.0
            else:
                anchor_image = max(images, key=lambda x: x["conf"])
                for image in images:
                    if image is anchor_image:
                        continue
                    # character_score += self.get_siglip_similarity(image["image_path"], anchor_image["image_path"])
                    character_score += self.fgclip2.get_image_similarity(image["image_path"], anchor_image["image_path"])
                character_score /= (len(images) - 1)
                consistency_score += character_score
        
        consistency_score /= len(character_obj_mapping)
        return consistency_score

    def extract_character_images(self, video_path: str, character_infos: dict, save_dir: str):
        """
        args:
            video_path: path to the tested video
            character_infos: information of all characters in the sample (current tested sample)
            save_dir: Directory to save the segmented images
            rep_k: We only assess "rep_k" images of certain segmented object to see if it's the character
            judge_thres: If no less than "judge_thres" images of the "rep_k" images of an object are 
                         recognized as the character by the MLLM, then it could be seen as the character
        return:
            concept_obj_images: image paths and confidence scores of all the segmented objects recognized by the sam3, organized into structured json data
                                {
                                    "concept_1": {
                                        "id_1": [
                                            {"image_path": xxx, "conf": xxx}
                                        ],
                                        ...
                                    }, 
                                    ...
                                }
            obj_character_mapping: like the following format, then you could easily get all images belonging to certain character
                                   {"character_name": f"{character_concept}_{object_id}", ...}

        """
        video_frames, _ = load_sampled_frames(video_path, step=3, offset=1)
        inference_session = self.sam3_processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )
        # 1. extract all images of certain concept (a character)
        concepts = list(set([character_info["sam_prompt"] for character_info in character_infos.values()]))
        concept_obj_images = {}

        inference_session = self.sam3_processor.add_text_prompt(
            inference_session=inference_session,
            text=concepts,
        )
        for concept in concepts:
            if concept not in concept_obj_images:
                concept_obj_images[concept] = {}

        for model_outputs in self.sam3.propagate_in_video_iterator(
            inference_session=inference_session, max_frame_num_to_track=len(video_frames)
        ):
            processed_outputs = self.sam3_processor.postprocess_outputs(inference_session, model_outputs)
            frame_idx = model_outputs.frame_idx
            
            frame = video_frames[frame_idx]
            H, W, _ = frame.shape
            
            # for i in range(len(outputs_per_frame)):
            frame_outputs = processed_outputs
            output_object_ids = frame_outputs['object_ids'].tolist()
            output_object_scores = frame_outputs['scores'].tolist()
            concept_to_ids = frame_outputs['prompt_to_obj_ids']
            for concept, object_ids in concept_to_ids.items():
                concept_dir = os.path.join(save_dir, concept)
                os.makedirs(concept_dir, exist_ok=True)
                
                for object_id in object_ids:
                    obj_id = str(object_id)

                    i = output_object_ids.index(object_id)
                    obj_conf = output_object_scores[i]
                    if obj_conf < CONF_THERSHOLD:
                        continue

                    object_dir = os.path.join(concept_dir, obj_id) 
                    os.makedirs(object_dir, exist_ok=True)
                    
                    mask = frame_outputs['masks'][i]
                    mask_np = mask.cpu().numpy().astype(bool)
                    
                    rgba = np.zeros((H, W, 4), dtype=np.uint8)
                    rgba[..., :3] = frame
                    rgba[..., 3] = mask_np.astype(np.uint8) * 255
                    
                    obj_frame_path = os.path.join(object_dir, f"frame_{frame_idx}.png")
                    # concept_obj_images[concept][obj_id].append(obj_frame_path)
                    if obj_id not in concept_obj_images[concept]:
                        concept_obj_images[concept][obj_id] = []

                    concept_obj_images[concept][obj_id].append({
                        "image_path": obj_frame_path,
                        "conf": obj_conf
                    })
                    imageio.imwrite(obj_frame_path, rgba)

            del model_outputs, processed_outputs

        # 2. attribute obj_id to character
        obj_character_mapping = {}
        for character_name, character_info in character_infos.items():
            character_concept = character_info["concept"]
            character_description = character_info["description"]
            sam_prompt = character_info["sam_prompt"]
            usr_prompt = CHARACTER_RECOGNITION_TEMP.format(character_concept=character_concept, character_description=character_description)
            # obj_images = concept_obj_images[character_concept]
            obj_images = concept_obj_images[sam_prompt]
            for obj_id, id_images in obj_images.items():
                # obj_key = f"{character_concept}_{obj_id}"
                obj_key = f"{sam_prompt}_{obj_id}"
                
                if obj_key in obj_character_mapping.values():
                    continue

                rep_image = max(id_images, key=lambda x: x["conf"])
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": rep_image["image_path"]},
                            {"type": "text", "text": usr_prompt}
                        ]
                    }
                ]
                judge_ans = self.get_mllm_response(messages)
                if "yes" in judge_ans.lower():
                    obj_character_mapping[character_name] = obj_key
                    break
            if character_name not in obj_character_mapping:
                obj_character_mapping[character_name] = ""

        return concept_obj_images, obj_character_mapping

    def get_mllm_response(self, messages: List[dict], max_new_tokens: int=128):
        inputs = self.mllm_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.mllm.device)
        # Inference: Generation of the output
        generated_ids = self.mllm.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.mllm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
        
    def load_sam3(self, model_name_or_path: str):
        self.sam3 = Sam3VideoModel.from_pretrained(model_name_or_path).to(self.device, dtype=torch.bfloat16)
        self.sam3_processor = Sam3VideoProcessor.from_pretrained(model_name_or_path)

    def load_mllm(self, model_name_or_path: str):
        self.mllm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.mllm_processor = AutoProcessor.from_pretrained(model_name_or_path)

    def load_fgclip2(self, model_name_or_path: str):
        self.fgclip2 = FgCLIP2(model_name_or_path)        


def get_args():
    parser = argparse.ArgumentParser(description="Overall video2text alignment evaluation launched.")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the list file of videos to be evaluated.")
    parser.add_argument("--sam3_path", type=str, default=f"{ROOT_ABS_PATH}/ckpts/sam3", help="Path to save the SAM3 used for semantic segmentation task.")
    parser.add_argument("--mllm_path", type=str, default=f"{ROOT_ABS_PATH}/ckpts/Qwen3-VL-8B-Instruct", help="Path to save the MLLM used for video QA task.")
    parser.add_argument("--clip_path", type=str, default=f"{ROOT_ABS_PATH}/ckpts/fg-clip2-base", help="Path to save the CLIP-based model used for image similarity computing.")
    parser.add_argument("--src_path", type=str, default=f"{ROOT_ABS_PATH}/data/prompt.json", help="Path to source the prompt metadata.")
    parser.add_argument("--temp_dir", type=str, default=f"{ROOT_ABS_PATH}/temp/character_consistency", help="Temporary directory to save the intermediate images genearted by sam3.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to save the evaluation results.")
    args = parser.parse_args()
    return args


def main(args):
    assert os.path.exists(args.eval_data), f"Failed to find corresponding file of {args.eval_data}, please carefully check your path again!"
    results = {
        "overall_score": 0.0,
        "sample_scores": {}
    }
    eval_name = os.path.splitext(os.path.basename(args.eval_data))[0]
    temp_dir = os.path.join(args.temp_dir, eval_name)
    os.makedirs(temp_dir, exist_ok=True)
    total_score = 0.0
    cc_scorer = CharConsistencyScorer(sam3_path=args.sam3_path, mllm_path=args.mllm_path, clip_path=args.clip_path)
    with open(args.eval_data, 'r') as f1, open(args.result_path, 'w') as f2, open(args.src_path, 'r') as f3:
        test_samples = json.load(f1)
        prompt_data = json.load(f3)
        for test_id, video_path in tqdm(test_samples.items()):
            theme_name = "_".join(test_id.split('_')[:-1])
            character_infos = prompt_data[theme_name][test_id]["character_infos"]
            with tempfile.TemporaryDirectory(dir=temp_dir) as tmp_dir:
                try:
                    sample_score = cc_scorer.compute_character_consistency(video_path=video_path, character_infos=character_infos, save_dir=tmp_dir)
                    results["sample_scores"][test_id] = sample_score
                    total_score += sample_score
                except Exception as e:
                    print(f"{str(e)} encountered when processing {video_path}! Ignoring this video...")
                    traceback.print_exc()
                    results["sample_scores"][test_id] = -1.0
        total_score = round(total_score / len(test_samples), 4)
        results["overall_score"] = total_score
        json.dump(results, f2, indent=4, ensure_ascii=False)
        print(f"Average character consistency score is: {total_score: .4f}")
    
    safe_rmtree(temp_dir)    


if __name__ == "__main__":
    args = get_args()
    main(args)