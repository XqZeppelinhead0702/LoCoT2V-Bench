import torch
from abc import ABC, abstractmethod
from typing import Union, List
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration
)


def load_images_videos(image_path: Union[str, List[str]]=None, video_path: Union[str, List[str]]=None):
    user_content = []
    if isinstance(image_path, list):
        for img_path in image_path:
            user_content.append({"type": "image", "image": img_path})
    elif isinstance(image_path, str):
        user_content.append({"type": "image", "image": image_path})
    else:
        pass
    if isinstance(video_path, list):
        for vid_path in video_path:
            user_content.append({"type": "video", "video": vid_path})
    elif isinstance(video_path, str):
        user_content.append({"type": "video", "video": video_path})
    else:
        pass
    return user_content


class MllmBase(ABC):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
    
    @abstractmethod
    def get_response(self,
        user_prompt: str, 
        system_prompt: str=None, 
        image_path: Union[str, List[str]]=None, 
        video_path: Union[str, List[str]]=None,
        **kwargs
    ):
        pass


class Qwen3VLModel(MllmBase):
    def __init__(self, model_name_or_path: str, torch_dtype=torch.bfloat16, attn_implementation: str="flash_attention_2", device_map: str="auto"):
        super().__init__(model_name_or_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

    def get_response(
        self,  
        user_prompt: str, 
        system_prompt: str=None, 
        image_path: Union[str, List[str]]=None, 
        video_path: Union[str, List[str]]=None,
        max_new_tokens: int=4096
    ):
        messages = []
        user_content = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content = load_images_videos(image_path, video_path)
        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()
    