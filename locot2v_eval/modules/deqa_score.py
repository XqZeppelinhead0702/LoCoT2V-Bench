import requests
import torch
from transformers import AutoModelForCausalLM
from PIL import Image


def load_deqa_model(model_name_or_path: str):
    print(f"Loading DeQA model from {model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        attn_implementation="eager", 
        torch_dtype=torch.float16,
        device_map="auto", 
    )
    return model


def get_deqa_score(model: AutoModelForCausalLM, image: str | Image.Image):
    if isinstance(image, Image.Image):
        score = model.score([image])
    else:
        score = model.score([Image.open(image)])
    return score.item()