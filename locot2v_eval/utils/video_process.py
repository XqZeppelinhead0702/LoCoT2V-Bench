import os
import numpy as np
import subprocess
import cv2
import random
import imageio
import ffmpeg

from typing import Optional, Callable, List, Tuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg


def min_frame_filter(scene_list: list, min_scene_frames: int=2):
    """
    filter scenes that have less than 3 frames
    """
    filtered = []
    for scene in scene_list:
        start, end = scene
        # get_frames() obtains the absolute frame number corresponding to the given time point
        duration_frames = end.get_frames() - start.get_frames()
        if duration_frames > min_scene_frames:
            filtered.append(scene)
        else:
            print(f"Drop out scenes that are too short: {duration_frames} frames")
    return filtered


def load_sampled_frames(video_path, step=2, offset=1):
    """
    Universal down-sampling function for loading video 
    :param video_path: Path to the video file
    :param step: Size of sampling window (e.g., step=3 means the size of each sampling window is 3)
    :param offset: Sampling offset (frame within the window, 0-indexed), (e.g., offset=1 means taking the second frame of the window)
    :return: frames list after sampled
    """
    frames = []
    sample_ids = []

    with imageio.get_reader(video_path, 'ffmpeg') as container:
        for i, frame in enumerate(container):
            if i % step == offset:
                frames.append(frame)
                sample_ids.append(i)
                
    return np.array(frames), sample_ids


def extract_all_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # output_pattern = os.path.join(output_dir, 'frame_%05d.jpg')
    output_pattern = os.path.join(output_dir, '%05d.jpg')
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '2',          
        output_pattern        
    ]
    subprocess.run(cmd, check=True)
    print(f"Frames saved to {output_dir}")
    
    
def pyscenedetect_split_videos_adaptive(
    video_path: str,
    segment_dir: str,
    step: float=0.5,
    max_times=1000,
    arg_override: str="-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 22 -c:a aac",
    show_progress: bool=True,
    show_output: bool=False,
    filter_func: Optional[Callable[[List[Tuple]], List[Tuple]]] = None
):
    # ensure best segmentation
    prev_list = []
    scene_list = []
    threshold = 0.0
    t = 0
    while (len(prev_list) == 0 or len(scene_list) > 0) and t < max_times:
        prev_list = scene_list
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        threshold += step
        t += 1
    
    scene_list = filter_func(prev_list) if filter_func is not None else prev_list
    if len(scene_list) > 0:
        split_video_ffmpeg(
            video_path,
            scene_list,
            output_dir=segment_dir,
            arg_override=arg_override,
            show_progress=show_progress,
            show_output=show_output
        )

    return scene_list