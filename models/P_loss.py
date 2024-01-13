import torch
from torchvision import models
from torch import nn
import numpy as np
import glob
import os


def load_model(model_name, model_dir):
    model = eval('models.%s(init_weights=False)' % model_name)
    path_format = os.path.join(model_dir, '%s-[a-z0-9]*.pth' % model_name)

    model_path = glob.glob(path_format)[0]
    model = model.cuda()

    model.load_state_dict(torch.load(model_path))
    for param in model.parameters():
        param.requires_grad = False
    return model


class Vgg19(torch.nn.Module):
    """ First layers of the VGG 19 model for the VGG loss.
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        model_path (str): Path to model weights file (.pth)
        requires_grad (bool): Enables or disables the "requires_grad" flag for all model parameters
    """

    def __init__(self, requires_grad: bool = False, vgg19_weights=None):
        super(Vgg19, self).__init__()
        if vgg19_weights is None:
            vgg_pretrained_features = load_model('vgg19', '../').features
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
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])#
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])#
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Vgg19_Unet(torch.nn.Module):

    def __init__(self, requires_grad: bool = False, vgg19_weights=None):
        super(Vgg19_Unet, self).__init__()
        if vgg19_weights is None:
            vgg_pretrained_features = load_model('vgg19', '../').features
        else:
            model = models.vgg19(pretrained=True)
            pretrain_dict = model.state_dict()
            layer1 = pretrain_dict['features.0.weight']
            # print(layer1.shape)
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


class PerceptualLoss(torch.nn.Module):
    """ Defines a criterion that captures the high frequency differences between two images.
    `"Perceptual Losses for Real-Time Style Transfer and Super-Resolution" <https://arxiv.org/pdf/1603.08155.pdf>`_
    Args:
        model_path (str): Path to model weights file (.pth)
    """
    def __init__(self, vgg19_weights=None):
        super(PerceptualLoss, self).__init__()
        self.vgg = Vgg19(vgg19_weights=vgg19_weights)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, vgg19_weights=None):
        super(ContrastiveLoss, self).__init__()
        self.vgg = Vgg19(vgg19_weights=vgg19_weights)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]


    def forward(self, negative, output, positive):
        n_vgg, p_vgg, o_vgg = self.vgg(negative), self.vgg(positive), self.vgg(output)
        loss = 0
        for i in range(len(o_vgg)):
            loss += self.weights[i] * self.criterion(o_vgg[i], p_vgg[i].detach())/(self.criterion(o_vgg[i], n_vgg[i])+self.criterion(o_vgg[i], n_vgg[i]))
        # print('contrastive loss',loss)
        return loss


class ContrastiveLoss_multiNegative(torch.nn.Module):
    """
    Contrastive loss with multiple negative samples
    """
    def __init__(self, vgg19_weights=None):
        super(ContrastiveLoss_multiNegative, self).__init__()
        self.vgg = Vgg19(vgg19_weights=vgg19_weights)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, negative_1, negative_2, negative_3, output, positive):
        n1_vgg, n2_vgg, n3_vgg, p_vgg, o_vgg = self.vgg(negative_1), self.vgg(negative_2), self.vgg(negative_3), self.vgg(positive), self.vgg(output)
        loss = 0
        for i in range(len(o_vgg)):
            loss += self.weights[i] * self.criterion(o_vgg[i], p_vgg[i].detach()) /\
                    (self.criterion(o_vgg[i], n1_vgg[i])+self.criterion(o_vgg[i], n2_vgg[i])+
                     +self.criterion(o_vgg[i], n3_vgg[i]))
        # print('contrastive loss',loss)
        return loss

