import os
import torch
import numpy as np
import torch.nn.functional as F

from typing import List
from PIL import Image
from torchvision import transforms
from torchvision.io import read_video
from decord import VideoReader, cpu


def change_fps(video_tensor, fps, target_fps):
    if target_fps == -1:
        return video_tensor
    
    # Obtain the frame rate and frame size of the input video
    num_frames, height, width, _ = video_tensor.shape

    # Calculate the sampling interval or interpolation interval
    if fps > target_fps:
        # Sampling interval
        interval = int(fps / target_fps)
        # Sample high-frame-rate video
        output_tensor = video_tensor[::interval]
    else:
        # Frame insertion interval
        interval = int(target_fps / fps)
        # Interpolate frames for low frame rate videos
        output_frames = []
        for frame in video_tensor:
            output_frames.append(frame)
            for _ in range(interval - 1):
                output_frames.append(frame)
        output_tensor = torch.stack(output_frames)
    return output_tensor


def resize_long_side(image_tensor, target_long_side):
    """
    Resizes a tensor image to have the specified size for its longest side while maintaining the aspect ratio.

    Args:
        image_tensor (torch.Tens
        or): Input image tensor of dtype torch.uint8 with shape (C, H, W).
        target_long_side (int): Desired size of the longest side of the resized image.

    Returns:
        torch.Tensor: The resized image tensor.
    """
    # Ensure that the input is a 3D tensor and the data type is uint8
    
    # Get the original dimensions of the image
    height, width = image_tensor.shape[-2:]

    # Determine whether the width or height is longer, and compute the new dimensions accordingly
    if width > height:
        scale = target_long_side / width
        new_width = target_long_side
        new_height = int(height * scale)
    else:
        scale = target_long_side / height
        new_height = target_long_side
        new_width = int(width * scale)

    # Reshape the image using bilinear interpolation (align_corners=False is recommended)
    resized_image = F.interpolate(
        image_tensor.float(),  
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False
    )  # Remove the batch dimension and convert back to uint8

    return resized_image


def get_video_data_and_length(video_paths: List[str], target_size: tuple=(224, 224)):
    all_video_frames = []
    video_lengths = []

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for path in video_paths:
        try:
            # 1. Load the frame by decord
            vr = VideoReader(path, ctx=cpu(0))
            
            # 2. Get indices of all the frames (partially sample the frames is also supported, like range(0, len(vr), 2))
            frame_indices = list(range(len(vr)))
            
            # 3. Load the frames and turn them into numpy array
            frames = vr.get_batch(frame_indices).asnumpy() # (T, H, W, C)
            
            # 4. Process each frame of the video
            processed_frames = [preprocess(frame) for frame in frames]
            
            # 5. Turns into Tensor: (T, 3, 224, 224)
            v_tensor = torch.stack(processed_frames)
            
            # 6. Merge data
            all_video_frames.append(v_tensor)
            video_lengths.append(len(frame_indices))
            
        except Exception as e:
            print(f"Warning：Failed to read the video {path}, pass. Error：{e}")
            continue
    if not all_video_frames:
        return None, []
    # 7. Concat all frames of the videos at the 0th dim
    video_data = torch.cat(all_video_frames, dim=0)
    
    return video_data, video_lengths


def calc_acf(features, k):
    acf = features[:-k] * features[k:]
    acf = acf.sum(-1).mean()
    return (1 - acf).abs().item()


def cal_segment(features):
    features = F.normalize(features, dim=-1, p=2)
    sims = features @ features.T
    # Obtain the upper triangular mask matrix
    mask = torch.triu(torch.ones_like(sims, dtype=torch.bool))

    # Use the mask to select the upper triangle elements
    sims = torch.masked_select(sims, mask)
    return sims.mean()


def get_segments(video, block_ratio):
    total_frames = video.shape[0]
    segment_length = int(total_frames * block_ratio)

    num_segments = int((total_frames + segment_length - 1) // segment_length)
    
    segments = []
    
    for i in range(num_segments):
        start = i * segment_length
        end = min(start + segment_length, total_frames)
        segment = video[start:end]
        
        if segment.size(0) < segment_length:
            repeats = segment_length // segment.size(0) + 1
            segment = torch.repeat_interleave(segment, repeats=repeats, dim=0)[:segment_length]
        
        segments.append(segment)
    
    return torch.stack(segments)


def to_same_size(segments, target_size=8):
    batch_size, segment_length, *rest_dims = segments.size()

    if segment_length > target_size:
        # Downsampling: Linear interpolation
        indices = torch.linspace(0, segment_length - 1, steps=target_size).long()
        resized_segments = segments[:, indices]
    else:
        # Upsampling: Filling by repeating the last frame
        repeats_needed = target_size - segment_length
        last_frame = segments[:, -1:]  # Take the last frame
        repeated_last_frames = last_frame.repeat(1, repeats_needed, *[1] * len(rest_dims))
        resized_segments = torch.cat((segments, repeated_last_frames), dim=1)

    return resized_segments