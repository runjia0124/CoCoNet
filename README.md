

  <h1 align="left">CoCoNet</h1>

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/IRFS/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9.0-%237732a8)](https://pytorch.org/)

Implementation of our work:

Jinyuan Liu\*, Runjia Lin\*, Guanyao Wu, Risheng Liu, Zhongxuan Luo, and Xin Fan<sup>📭</sup>, "**CoCoNet: Coupled Contrastive Learning Network with Multi-level Feature Ensemble for Multi-modality Image Fusion**", International Journal of Computer Vision (**IJCV**), 2023.


#### [[Paper](https://link.springer.com/article/10.1007/s11263-023-01952-1)]    [[Arxiv](https://arxiv.org/pdf/2211.10960.pdf)]


## Introduction

We present CoCoNet, the first work to explore contrastive prior for multi-modality image fusion. 

![](demo/pipeline.png)


- Check out our recent related works 🆕:
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

## Testing
```
python main.py \
--test --use_gpu \    
--test_vis ./TNO/VIS \
--test_ir ./TNO/IR 
```
To work with your own test set, make sure to use the same file names for each infrared-visible image pair if you prefer not to edit the code.  

## Training
Coming soon...

## Results
### Visual inspection
![](demo/visual.png)
### Down-stream task
![](demo/visual_2.png)



## Contact
The training code is being sorted, if you have any questions about the testing code, please email us or open an issue, 

Runjia Lin(`linrunja@gmail.com`) or Jinyuan Liu (`atlantis918@hotmail.com`).

## Citation

If you find this paper/code helpful, please consider citing us: 

```
@article{liu2023coconet,
  title={Coconet: Coupled contrastive learning network with multi-level feature ensemble for multi-modality image fusion},
  author={Liu, Jinyuan and Lin, Runjia and Wu, Guanyao and Liu, Risheng and Luo, Zhongxuan and Fan, Xin},
  journal={International Journal of Computer Vision},
  pages={1--28},
  year={2023},
  publisher={Springer}
}
```

