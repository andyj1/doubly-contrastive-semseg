## Doubly Contrastive Semantic Segmentation

[Project Page](https://bmvc2022.mpi-inf.mpg.de/460/)
[PDF](https://bmvc2022.mpi-inf.mpg.de/0460.pdf)
[Poster](https://bmvc2022.mpi-inf.mpg.de/0460_poster.pdf)

- PyTorch code for the paper: Doubly Contrastive End-to-End Semantic Segmentation for Autonomous Driving under Adverse Weather

### Abstract
Road scene understanding tasks have recently become crucial for self-driving vehicles. In particular, real-time semantic segmentation is indispensable for intelligent self-driving agents to recognize roadside objects in the driving area. As prior research works have primarily sought to improve the segmentation performance with computationally heavy operations, they require far significant hardware resources for both training and deployment, and thus are not suitable for real-time applications. As such, we propose a doubly contrastive approach to improve the performance of a more practical lightweight model for self-driving, specifically under adverse weather conditions such as fog, nighttime, rain and snow. Our proposed approach exploits both image- and pixel-level contrasts in an end-to-end supervised learning scheme without requiring a memory bank for global consistency or the pretraining step used in conventional contrastive methods. We validate the effectiveness of our method using SwiftNet on the ACDC dataset, where it achieves up to 1.34%p improvement in mIoU (ResNet-18 backbone) at 66.7 FPS (2048×1024 resolution) on a single RTX 3080 Mobile GPU at inference. Furthermore, we demonstrate that replacing image-level supervision with self-supervision achieves comparable performance when pre-trained with clear weather images.

### Acknowledgments

- This  work  was  supported  by  the Institute  for  Information  &  Communications  Technology  Promotion  (IITP)grant funded by the Korea government (MSIT) (No.2020-0-00440, Development of Artificial Intelligence Technology that Continuously Improves Itself as the Situation Changes in the Real World).

### Citation
```bash
@inproceedings{Jeong_2022_BMVC,
author    = {Jongoh Jeong and Jong-Hwan Kim},
title     = {Doubly Contrastive End-to-End Semantic Segmentation for Autonomous Driving under Adverse Weather},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0460.pdf}
}
```