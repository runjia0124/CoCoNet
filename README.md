

<div align="center">
  <h1 align="center">CoCoNet: Coupled Contrastive Learning Network with Multi-level Feature Ensemble for Multi-modality Image Fusion</h1>

  <p align="center">
    <a href="https://github.com/JinyuanLiu-CV">Jinyuan Liu</a><sup>1</sup>,
	<a href=https://github.com/runjia0124/ target=_blank rel=noopener>Runjia Lin</a><sup>1</sup>, 
    Guanyao Wu<sup>1</sup>, 
    <a href=https://rsliu.tech/ target=_blank rel=noopener>Risheng Liu</a><sup>1,2</sup>,
    Zhongxuan Luo<sup>1</sup> and 
    Xin Fan<sup>1</sup>
    </sup>
      <br>
      <sup>1</sup>Dalian University of Technology
      <sup>2</sup>Peng Cheng Laboratory
    <br />
    <a href="https://arxiv.org/abs/2203.03949">Arxiv</a> 

  </p>
</div>

## Introduction

we propose a coupled contrastive learning network, dubbed CoCoNet, to realize infrared and visible image fusion in an end-to-end manner. Concretely, to simultaneously retain typical features from both modalities and remove unwanted information emerging on the fused result, we develop a coupled contrastive constraint in our loss this http URL a fused imge, its foreground target/background detail part is pulled close to the infrared/visible source and pushed far away from the visible/infrared source in the representation space. We further exploit image characteristics to provide data-sensitive weights, which allows our loss function to build a more reliable relationship with source images. Furthermore, to learn rich hierarchical feature representation and comprehensively transfer features in the fusion process, a multi-level attention module is established. In addition, we also apply the proposed CoCoNet on medical image fusion of different types, e.g., magnetic resonance image and positron emission tomography image, magnetic resonance image and single photon emission computed tomography image. Extensive experiments demonstrate that our method achieves the state-of-the-art (SOTA) performance under both subjective and objective evaluation, especially in preserving prominent targets and recovering vital textural details.

![](demo/pipeline.png)

## Installation

Clone repo:
```
git clone https://github.com/runjia0124/CoCoNet.git
cd CoCoNet
```

The code is tested with Python == 3.7, PyTorch == 1.10.1 and CUDA == 11.3 on NVIDIA GeForce RTX 3090. We recommend you to use [anaconda](https://www.anaconda.com/) to manage dependencies. You may need to change the torch and cuda version in the `requirements.txt` according to your computer.
```
conda create -n rcmvsnet python=3.7
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda activate rcmvsnet
pip install -r requirements.txt
```

## Datasets

### DTU

**Training**

Download the [DTU dataset](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) pre-processed by [MVSNet](https://github.com/YoYo000/MVSNet) and extract the archive. You could use [gdown](https://github.com/wkentaro/gdown) to download it form Google Drive. You could refer to [MVSNet](https://github.com/YoYo000/MVSNet) for the detailed documents of the file formats.

Download the original resolution [depth maps](https://drive.google.com/open?id=1LVy8tsWajG3uPTCYPSxDvVXFCdIYXaS-) provided by [YaoYao](https://github.com/YoYo000/MVSNet/issues/106). Extract it and rename the folder to `Depths_raw`. 

Merge the folders together and you should get a dataset folder like below:

```
dtu
├── Cameras
├── Depths
├── Depths_raw
└── Rectified
```

**Testing**

Download the [DTU testing dataset](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view) pre-processed by [MVSNet](https://github.com/YoYo000/MVSNet) and extract the archive. You could use [gdown](https://github.com/wkentaro/gdown) to download it form Google Drive. You could refer to [MVSNet](https://github.com/YoYo000/MVSNet) for the detailed documents of the file formats. 

```
dtu_test
├── scan1
├── scan4
├── scan9
...
├── scan114
└── scan118
```



### Tanksandtemples(Only for Testing)

Download the [Tanks and Temples testing set](https://drive.google.com/open?id=1YArOJaX9WVLJh4757uE8AEREYkgszrCo) pre-processed by [MVSNet](https://github.com/YoYo000/MVSNet). For the `intermediate` subset, remember to replace the cameras by those in `short_range_caemeras_for_mvsnet.zip` in the `intermediate` folder, see [here](https://github.com/YoYo000/MVSNet/issues/14). You should get a dataset folder like below:

```
tankandtemples
├── advanced
│   ├── Auditorium
│   ├── Ballroom
│   ├── Courtroom
│   ├── Museum
│   ├── Palace
│   └── Temple
└── intermediate
    ├── Family
    ├── Francis
    ├── Horse
    ├── Lighthouse
    ├── M60
    ├── Panther
    ├── Playground
    └── Train
```

## Configure

There are several options of flags at the beginning of each train/test file. Several key options are explained below. Other options are self-explanatory in the codes. Before running our codes, you may need to change the `true_gpu`, `trainpath/testpath` , `logdir`and `loadckpt` (only for testing).

* `logdir` A relative or absolute folder path for writing logs.
* `true_gpu` The true GPU IDs, used for setting CUDA_VISIBLE_DEVICES in the code. You may change it to your GPU IDs.
* `gpu` The GPU ID used in your experiment. If true_gpu: "5, 6". Then you could use gpu: [0], gpu: [1], or gpu: [0, 1]
* `loadckpt` The checkpoint file path used for testing.
* `trainpath/testpath` A relative or absolute folder path for training or testing data. You may need to change it to your data folder.
* `outdir` A relative or absolute folder path for generating depth maps and writing point clouds(DTU).
* `plydir` A relative or absolute folder path for writing point clouds(Tanks).
* `dataset` Dataset to be used. ["dtu_train","dtu_test","tanks"]
* `resume` Resume training from the latest history.

## Training

Train the model on DTU dataset
```
python train_rcmvsnet.py --logdir ./rc-mvsnet --trainpath {your data dir} --dataset dtu_train --gpu [0,1,2,3] --true_gpu 0,1,2,3 
```

## Testing

### **DTU**

We have provided pre-trained model in the `pretrain` folder, which contains models for both backbone network and rendering consistency network, only the backbone network (ended with 'cas') is used for testing as mentioned in the paper. The rendering consistency network (ended with 'nerf') is used for resume training from the current epoch. 

You could use `eval_rcmvsnet_dtu.py` to reconstruct depthmaps and point clouds with the checkpoint. To reproduce the DTU results in our paper, run commands below:

```
python eval_rcmvsnet_dtu.py
```
After you get the point clouds, you could follow the instructions in [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36) website to quantitatively evaluate the point clouds.

**DTU Point Cloud Evaluation**

We provide evaluation code in the `matlab_eval` folder. The code relies on the official code of [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36) Dataset. Please use  `BaseEvalMain_web_pt.m`, `ComputeStat_web_pt.m` and `compute_mean.m` for evaluation. 

* `gt_datapath` The path to the ground truth point clouds.
* `dataPaths` The path to the generated point clouds of RC-MVSNet.
* `resultsPaths` The path to output metrics of the evaluation script.

### Tanksandtemples

To reproduce the Tanksandtemples results in our paper, run commands below:
```
python eval_rcmvsnet_tanks.py --split "intermediate" --loadckpt "./pretrain/model_000014_cas.ckpt"  --plydir "./tanks_submission" --outdir './tanks_exp' --testpath {your data dir}
```
```
python eval_rcmvsnet_tanks.py --split "advanced"  --loadckpt "./pretrain/model_000014_cas.ckpt" --plydir "./tanks_submission" --outdir './tanks_exp' --testpath {your data dir}
```
After you get the point clouds, you could submit them to the [Tanksandtemples](https://www.tanksandtemples.org/) website for quantitative evaluatation.


## Contact

If you have any questions, please raise an issue or email to Runjia Lin (`linrunja@gmail.com`) or Jinyuan Liu (` atlantis918@hotmail.com`).

## Acknowledgments

Template is adapted from this awesome repository. Appreciate!

* [RC-MVSNet](https://github.com/Boese0601/RC-MVSNet)
