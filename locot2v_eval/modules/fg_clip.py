import torch
from typing import List
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)


def determine_max_value(image):
    w,h = image.size
    max_val = (w//16)*(h//16)
    if max_val > 784:
        return 1024
    elif max_val > 576:
        return 784
    elif max_val > 256:
        return 576
    elif max_val > 128:
        return 256
    else:
        return 128
    
    
class FgCLIP2:
    def __init__(self, model_name_or_path: str, device: str="cuda"):
        if device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).cpu()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    
    def get_image_similarity(self, image1: str | Image.Image, image2: str | Image.Image):
        image1_feature = self.get_norm_image_feature(image1)
        image2_feature = self.get_norm_image_feature(image2)
        
        similarity = image1_feature @ image2_feature.T
        similarity = similarity.cpu()

        return similarity.item()
    
    def get_norm_image_feature(self, image: str | Image.Image):
        if isinstance(image, str):
            image_obj = Image.open(image).convert("RGB")
        else:
            image_obj = image            
        image_input = self.image_processor(images=image_obj, max_num_patches=determine_max_value(image_obj), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            image_feature = self.model.get_image_features(**image_input)
            image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
        
        return image_feature

    def get_norm_text_feature(self, text: str | List[str], walk_type="box"):
        captions = []
        if isinstance(text, str):
            captions.append(text.lower())
        else:
            captions = [t.lower() for t in text]
        with torch.no_grad():
            caption_input = self.tokenizer(captions, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.model.device)
            text_feature = self.model.get_text_features(**caption_input, walk_type=walk_type)
            text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
        
        return text_feature