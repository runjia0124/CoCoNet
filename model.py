import time
import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from options.train_options import TrainOptions
from torch.utils.tensorboard import SummaryWriter
from lib.nn import SynchronizedBatchNorm2d as SynBN2d
from utils import pad_tensor
from utils import pad_tensor_back
import torchfile
from torchvision import models
from resnet import resnet18
from attention import CAM_Module
from P_loss import Vgg19_Unet, Vgg19_train

class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.ReLU1_1 = nn.ReLU()
        self.bn1_1 = nn.BatchNorm2d(64)
        self.max_pool_1_1 = nn.MaxPool2d(2, 2)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.ReLU1_2 = nn.ReLU()
        self.bn1_2 = nn.BatchNorm2d(64)
        self.max_pool_1_2 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.ReLU2_1 = nn.ReLU()
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.ReLU2_2 = nn.ReLU()
        self.bn2_2 = nn.BatchNorm2d(128)
        # self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)
        # self.ReLU2_3 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.ReLU3_1 = nn.ReLU()
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.ReLU3_2 = nn.ReLU()
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.ReLU3_3 = nn.ReLU()
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.ReLU3_4 = nn.ReLU()
        self.bn3_4 = nn.BatchNorm2d(256)
        # self.max_pool_3 = nn.MaxPool2d(2, 2)
    def forward(self, input):
        o_64 = self.bn1_1(self.ReLU1_1(self.conv1_1(input)))
        o_64 = self.max_pool_1_1(o_64)
        o_64 = self.bn1_2(self.ReLU1_2(self.conv1_2(o_64)))
        x = self.max_pool_1_2(o_64)

        o_128 = self.bn2_1(self.ReLU2_1(self.conv2_1(x)))
        o_128 = self.bn2_2(self.ReLU2_2(self.conv2_2(o_128)))
        x = self.max_pool_2(o_128)

        o_256 = self.bn3_1(self.ReLU3_1(self.conv3_1(x)))
        o_256 = self.bn3_2(self.ReLU3_2(self.conv3_2(o_256)))
        o_256 = self.bn3_3(self.ReLU3_3(self.conv3_3(o_256)))
        o_256 = self.bn3_4(self.ReLU3_4(self.conv3_4(o_256)))
        return o_64, o_128, o_256
        

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.expand_as(x).shape)
        # print(y.expand_as(x))
        return x * y.expand_as(x), y.expand_as(x)


class Unet_resize_conv(nn.Module):
    def __init__(self):
        super(Unet_resize_conv, self).__init__()

        # self.vgg = load_vgg16("./core/Losses/vgg", cfg)
        # self.vgg.to_relu_1_2[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.se_32 = CAM_Module(32)
        self.se_64 = CAM_Module(64)
        self.se_128 = CAM_Module(128)
        self.se_256 = CAM_Module(256)

        self.vgg19 = Vgg19_Unet(vgg19_weights='place_holder')


        self.skip = False
        p = 1
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)

        self.conv1_1 = nn.Conv2d(2, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.deconv4 = nn.Conv2d(256 * 3, 256, 3, padding=p)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)

   
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.att_deconv7 = nn.Conv2d(128*3, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.att_deconv8 = nn.Conv2d(64 * 3, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.att_deconv9 = nn.Conv2d(32 * 3, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 1, 1)

        self.tanh = nn.Tanh()

    def forward(self, ir, vis):
        maps = []

        input = torch.cat([ir, vis], 1)
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        vis, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(vis)
        ir, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(ir)

        vis_2, vis_3, vis_4 = self.vgg19(vis)
        ir_2, ir_3, ir_4 = self.vgg19(ir)


        flag = 0

        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))  # 256
        unet_out, unet_map = self.se_256(x)
        vgg_v_out, vgg_v_map = self.se_256(vis_4)
        vgg_i_out, vgg_i_map = self.se_256(ir_4)

        att_4 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        x = self.deconv4(att_4)  # 256*3 -> 256, deconv the concated attention maps
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))

        conv6 = F.upsample(conv4, scale_factor=2, mode='bilinear')  # 256

        unet_out, unet_map = self.se_128(conv3) # conv3 128
        vgg_v_out, vgg_v_map = self.se_128(vis_3)
        vgg_i_out, vgg_i_map = self.se_128(ir_3)

        att_7 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        att_7 = self.att_deconv7(att_7) # 128*3 -> 128
        up7 = torch.cat([self.deconv6(conv6), att_7], 1) # deconv6, 256->128
        # up7 = self.deconv6(torch.cat([conv6, vis_c7, ir_c7], 1))  # 256 + 256 + 256 -> 256
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))  #
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))  # 128

        unet_out, unet_map = self.se_64(conv2)
        vgg_v_out, vgg_v_map = self.se_64(vis_2)
        vgg_i_out, vgg_i_map = self.se_64(ir_2)
        maps.append(conv2)
        maps.append(unet_out)

        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')

        att_8 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        att_8 = self.att_deconv8(att_8) # 64*3 -> 64

        up8 = torch.cat([self.deconv7(conv7), att_8], 1) # deconv7, 128->64
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))  

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')

        up9 = torch.cat([self.deconv8(conv8), conv1], 1) # deconv8, 64 -> 32
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)

        latent = self.tanh(latent)
        # latent = (latent + 1) / 2
        output = latent

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        # gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output
            

if __name__ == '__main__':
    model = Resnet_18()







