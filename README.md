

## Installation

Clone repo:
```
git clone https://github.com/runjia0124/CoCoNet.git
cd CoCoNet
```

The code is tested with Python == 3.7, PyTorch == 1.8.1 and CUDA == 11.1 on NVIDIA GeForce RTX 2080. 
```
conda create -n coconet python=3.7
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c pytorch -c conda-forge
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

## Training
Coming soon...


## Contact
Training code is not yet well sorted (will do), any question about the code, please email us or open an issue, 

Jinyuan Liu (`atlantis918@hotmail.com`) or Runjia(`linrunja@gmail.com`).

## Acknowledgments

Template is adapted from this awesome repository. Appreciate!

* [RC-MVSNet](https://github.com/Boese0601/RC-MVSNet)
