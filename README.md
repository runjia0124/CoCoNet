

  <h1 align="left">CoCoNet</h1>

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/IRFS/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9.0-%237732a8)](https://pytorch.org/)

Implementation for our paper:

>  CoCoNet: Coupled Contrastive Learning Network with Multi-level Feature Ensemble for Multi-modality Image Fusion
>
>  [Jinyuan Liu\*](https://scholar.google.com/citations?user=a1xipwYAAAAJ&hl=zh-CN&oi=ao), Runjia Lin\*, Guanyao Wu, Risheng Liu, Zhongxuan Luo, and Xin Fan
>
>  International Journal of Computer Vision (IJCV), 2023

### [Arxiv](https://arxiv.org/pdf/2211.10960.pdf)


## Introduction

We propose a coupled contrastive learning network, dubbed CoCoNet, to realize infrared and visible image fusion in an end-to-end manner. Concretely, to simultaneously retain typical features from both modalities and remove unwanted information emerging on the fused result, we develop a coupled contrastive constraint in our loss this http URL a fused imge, its foreground target/background detail part is pulled close to the infrared/visible source and pushed far away from the visible/infrared source in the representation space. We further exploit image characteristics to provide data-sensitive weights, which allows our loss function to build a more reliable relationship with source images. Furthermore, to learn rich hierarchical feature representation and comprehensively transfer features in the fusion process, a multi-level attention module is established. In addition, we also apply the proposed CoCoNet on medical image fusion of different types, e.g., magnetic resonance image and positron emission tomography image, magnetic resonance image and single photon emission computed tomography image. Extensive experiments demonstrate that our method achieves the state-of-the-art (SOTA) performance under both subjective and objective evaluation, especially in preserving prominent targets and recovering vital textural details.

![](demo/pipeline.png)


- Related works
  - 🔥 **ICCV'23 Oral:** Multi-interactive Feature Learning and a Full-time Multi-modality Benchmark for Image Fusion and Segmentation [[paper]](https://arxiv.org/pdf/2308.02097.pdf) [[code]](https://github.com/JinyuanLiu-CV/SegMiF)
     
  - 🔥 **CVPR'22 Oral:** Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality
Benchmark to Fuse Infrared and Visible for Object Detection [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.pdf) [[code]](https://github.com/JinyuanLiu-CV/TarDAL)
  - 🔥 **IJCAI'23:** Bi-level Dynamic Learning for Jointly Multi-modality Image Fusion and Beyond [[paper]](https://arxiv.org/pdf/2305.06720.pdf) [[code]](https://github.com/LiuZhu-CV/BDLFusion)



## Installation

Clone repo:
```
git clone https://github.com/runjia0124/CoCoNet.git
cd CoCoNet
```

The code is tested with Python == 3.8, PyTorch == 1.9.0 and CUDA == 11.1 on NVIDIA GeForce RTX 2080, you may use a different version according to your GPU. 
```
conda create -n coconet python=3.8
conda activate coconet
pip install -r requirements.txt
```

## Results
### Visual inspection
![](demo/visual.png)
### Down-stream task
![](demo/visual_2.png)

## Testing
```
python main.py \
--test --use_gpu \    
--test_vis ./TNO/VIS \
--test_ir ./TNO/IR 
```

## Training
Coming soon...


## Contact
The training code is being sorted, if you have any questions about the testing code, please email us or open an issue, 

Runjia Lin(`linrunja@gmail.com`) or Jinyuan Liu (`atlantis918@hotmail.com`).

## Acknowledgments

Template is adapted from this awesome repository. Appreciate!

* [RC-MVSNet](https://github.com/Boese0601/RC-MVSNet)

