from torchstat import stat
from model import Unet_resize_conv, Unet_measure
import torch

model = Unet_measure()
stat(model, (2, 620, 448))