#!/bin/bash
set -e

export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_ENABLE_HF_TRANSFER=1

# Huggingface Download List

## zhiyuanyou/DeQA-Score-Mix3
huggingface-cli download --resume-download --local-dir-use-symlinks False zhiyuanyou/DeQA-Score-Mix3 --local-dir ./DeQA-Score-Mix3

## Qwen/Qwen3-VL-8B-Instruct
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen3-VL-8B-Instruct --local-dir ./Qwen3-VL-8B-Instruct

## Qwen/Qwen3-VL-4B-Instruct
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen3-VL-4B-Instruct --local-dir ./Qwen3-VL-4B-Instruct

## facebook/sam3
huggingface-cli download --resume-download --local-dir-use-symlinks False facebook/sam3 --local-dir ./sam3

## qihoo360/fg-clip2-base
huggingface-cli download --resume-download --local-dir-use-symlinks False qihoo360/fg-clip2-base --local-dir ./fg-clip2-base


# Directly Download
# If you fail to download these checkpoints, please download them locally and then upload these ckpts to your remote server

## RAFT
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
mv ./models ./RAFT

## AMT
mkdir -p ./AMT
wget https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth -O ./AMT/amt-s.pth
wget https://github.com/MCG-NKU/AMT/blob/main/cfgs/AMT-S.yaml -O ./AMT/AMT-S.yaml

## devil weights
mkdir -p ./devil_weights
