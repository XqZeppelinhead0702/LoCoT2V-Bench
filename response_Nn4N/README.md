# Complementary Results
## [W2/Q2: Differentiation from VBench-Long]
The comparison between VBench-Long and LoCoT2V-Bench based on correlation to human evaluation results including PLCC, SRCC and KRCC are as follows:
Note that for VBench-Long we use the average of its aesthetic quality and imaging quality as perceptual quality, subject consistency as character consistency.

| Dimension                 |      Perceptual Quality       |       Overall Alignment       | Subject/Character Consistency |    Background Consistency     |
| ------------------------- | :---------------------------: | :---------------------------: | :---------------------------: | :---------------------------: |
| VBench-Long               |       55.10/50.69/36.77       |       10.99/18.18/12.04       |       41.09/32.65/22.26       |       45.06/35.38/17.53       |
| **LoCoT2V-Bench (ours.)** | **71.39**/**70.20**/**54.67** | **63.38**/**67.23**/**52.68** | **47.17**/**49.90**/**39.19** | **52.80**/**51.11**/**36.57** |

## [W3: Lack of Important Details]
See some of our filtered videos in [./filtered_video_cases](./filtered_video_cases). We recommend the reviewer downloading them to watch locally for better visualization.
Although more representative videos may have been deleted, we still expect these videos could serve as a partial reference for our filtering criteria.

## [W4/Q3: Lack Evaluation on Commercial Methods and Real-world Videos]
We conduct experiments on the same selected 10 samples of commercial API evaluation and demonstrate the results as follows:

| Model            |  PQ   | TVA-OA | TVA-FGA | TVA-Avg. | TQ-CC | TQ-BC | TQ-WE | TQ-Avg. | HERD  |  DQ   | Avg.  |
| ---------------- | :---: | :----: | :-----: | :------: | :---: | :---: | :---: | :-----: | :---: | :---: | :---: |
| Sora2            | 68.73 | **80.00**  |  53.45  |  66.73   | **56.55** | **99.32** | 98.99 |  **84.95**  | **92.33** | 66.84 | **75.92** |
| Seedance 1.5 pro | 71.51 | 73.00  |  46.30  |  59.65   | 40.86 | 99.27 | 98.00 |  79.38  | 89.00 | 57.85 | 71.48 |
| Seedance 2.0     | 76.43 | 79.00  |  55.86  |  **67.43**   | 38.93 | 99.11 | 97.60 |  78.55  | 85.67 | 57.20 | 73.05 |
| Kling 3.0        | **76.50** | 71.00  |  **60.47**  |  65.74   | 22.32 | 99.26 | **99.79** |  73.78  | 87.33 | 57.62 | 72.19 |
| Real-world Video | 69.40 | 53.00  |  48.47  |  50.74   | 9.47  | 99.11 | 98.95 |  69.17  | 78.67 | **67.17** | 67.03 |

## [W5/Q4: Disentanglement of Video Duration and Prompt Complexity]
We provide results of the comparison results between Long+Complex and Short+Complex across all dimensions as follows:
|Model|PQ|TVA-OA|TVA-FGA|TVA-Avg.|TQ-CC|TQ-BC|TQ-WE|TQ-Avg.|HERD|DQ|Avg.|
|-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|CausVid|68.82|**47.78**|**26.30**|**37.04**|**46.58**|99.00|95.92|**80.50**|72.41|**58.43**|**63.44**|
| ~short-complex |**76.01**|42.59|21.45|32.02|42.84|**99.06**|**96.38**|79.43|**73.09**|51.04|62.32|
|LongLive|80.12|**58.89**|**38.34**|**48.62**|**57.46**|**99.22**|97.09|**84.59**|84.01|**62.11**|**71.89**|
| ~short-complex |**81.53**|51.20|31.17|41.19|48.15|99.21|**97.32**|81.56|**84.57**|53.89|68.55|
