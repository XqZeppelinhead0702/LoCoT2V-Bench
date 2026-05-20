# LoCoT2V-Bench
[![arXiv:2510.26412](https://img.shields.io/badge/arXiv-2510.26412-red?logo=arXiv:2510.26412)](https://arxiv.org/abs/2510.26412)

Official implementation of **LoCoT2V-Bench: Benchmarking Long-Form and Complex Text-to-Video Generation** (ICML 2026)

If you have any questions, feel free to contact us with this Email: [xiangqingzheng0702@gmail.com](xiangqingzheng0702@gmail.com)

🚩: TODOs
> - [x] release the prompt data and metadata for evaluation
> - [ ] release the source videos (will be a link to download these videos)
> - [ ] release evaluation code
> - [ ] release the final version paper on arxiv

## Summary
LoCoT2V-Bench is a comprehensive evaluation benchmark designed to assess long-form text-to-video generation under complex, multi-scene textual conditions. Grounded in high-quality real-world videos, the benchmark provides detailed, script-level prompts enriched with hierarchical metadata, including specific character attributes and camera movements. To systematically measure generation quality, we introduce LoCoT2V-Eval, a multi-dimensional evaluation framework that assesses fine-grained text-video alignment through a hierarchical tree-structured VQA approach and captures high-level narrative fulfillment via a dual-agent Human Expectation Realization Degree (HERD) metric. Extensive evaluations across 17 leading models reveal that while current approaches excel in overall visual quality and background stability, they still face significant challenges in fine-grained semantic adherence and long-term character consistency.

## Data Introduction
To be completed...

(See [data/prompt.json](./data/prompt.json) for more details) We've provided our prompt data for generating videos and metadata for evaluation. Here we give a brief explanation for the fields in it.
* **[theme] (like "space")**: Corresponding to the 18 themes mentioned in our paper.
* **[theme]_[id] (like space_2)**: Refer to the id we assigned to our collected source video.
* **source_description**: The description about the content of the source video.
* **frame_num_ref**: The number of frames of the source video.
* **duration_ref**: The duration of the source videos (seconds).
* **prompt**: The prompt we used to generate videos for subsequent evaluation.
* **purified prompt**: Purified version of the prompt after removing character with more abstract description (such as "Aria Lorne"->"a woman with short silver hair..."), which is better for leveraging MLLM to assign scores on Overall Text-Video Alignment (MLLMs can't know who the character is based on his/her/its name).
* **character_infos**: Necessary informations for each character occurring in the prompt.
  * *concept*: What is the character? (a man? a dog?)
  * *description*: Detailed description of the character. 
  * *attributes*: Finer-grained attributes of the character. (Used in our Fine-grained Alignment Evaluation)
  * *sam_prompt*: Simpler description for SAM3 extraction.
* **background_infos**: ...
* **scene_infos**: ...
* **fga_questions**: ...
* **herd_expectations**: ...
* **herd_questions**: ...

## Enviroment Setting
...

## Evaluation Guidance
...

## Citation
If you find our work helpful for your research, please consider citing our work.
```
@article{zheng2025locot2v,
  title={LoCoT2V-Bench: Benchmarking Long-Form and Complex Text-to-Video Generation},
  author={Zheng, Xiangqing and Wu, Chengyue and Chen, Kehai and Zhang, Min},
  journal={arXiv preprint arXiv:2510.26412},
  year={2025}
}
```
