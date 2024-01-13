import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import pad_tensor
from utils import pad_tensor_back
from utils.attention import CAM_Module

class Vgg19_Unet(torch.nn.Module):

    def __init__(self, requires_grad: bool = False, vgg19_weights=None):
        super(Vgg19_Unet, self).__init__()
        if vgg19_weights is None:
            vgg_pretrained_features = load_model('vgg19', './').features
        else:
            model = models.vgg19(pretrained=True)
            pretrain_dict = model.state_dict()
            layer1 = pretrain_dict['features.0.weight']
            new = torch.zeros(64, 1, 3, 3)
            for i, output_channel in enumerate(layer1):
                # Grey = 0.299R + 0.587G + 0.114B, RGB2GREY
                new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
            pretrain_dict['features.0.weight'] = new
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            model.load_state_dict(pretrain_dict)
            vgg_pretrained_features = model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        self.slice1.add_module(str(2), nn.MaxPool2d(2, 2))
        for x in range(2, 4):
            self.slice1.add_module(str(x+1), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x+1), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x+1), vgg_pretrained_features[x])#

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):

        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)

        return h_relu1, h_relu2, h_relu3


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

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.att_deconv8 = nn.Conv2d(64 * 3, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64)

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
        att_7 = self.att_deconv7(att_7)  # 128*3 -> 128
        up7 = torch.cat([self.deconv6(conv6), att_7], 1)  # deconv6, 256->128
        # up7 = self.deconv6(torch.cat([conv6, vis_c7, ir_c7], 1))  # 256 + 256 + 256 -> 256
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))  
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))  

        unet_out, unet_map = self.se_64(conv2)
        vgg_v_out, vgg_v_map = self.se_64(vis_2)
        vgg_i_out, vgg_i_map = self.se_64(ir_2)
        maps.append(conv2)
        maps.append(unet_out)

        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')

        att_8 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        att_8 = self.att_deconv8(att_8) # 64*3 -> 64

        up8 = torch.cat([self.deconv7(conv7), att_8], 1)  # deconv7, 128->64
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))  

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')

        up9 = torch.cat([self.deconv8(conv8), conv1], 1)  # deconv8, 64 -> 32
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)

        latent = self.tanh(latent)
        output = latent

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)

        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output








