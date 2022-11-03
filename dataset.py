import os

import imageio
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import h5py
import cv2


# v i
# o u
# 
class TrainDataSet(Dataset):
    def __init__(self, dataset=None, arg=None):
        super(TrainDataSet, self).__init__()
        self.arg = arg
        # self.transform = transform

        self.source_data = []
        self.grad, self.en = [], []
        self.batch_size = 1
        # self.task_num = len(dataset)
        len_ = []
        # for i in range(self.task_num):
        data = h5py.File(dataset, 'r')
        # grad = data['grad']
        # en = data['mean']

        # dataset from guanyao
        keys = list(data.keys())
        print('sample num: ', len(keys))
        # print('keys: ', keys)
        print('data: ', data['0'].shape)
        # print('data type: ', type(data['0']))
        # print('data: ', data['0'][...].shape)
        # print('data min: ', data['2'][...].min())
        # print('data max: ', data['2'][...].max())
        # imageio.imsave('./c1.png', data['1130'][...][..., 0])
        # imageio.imsave('./c2.png', data['1130'][...][..., 1])
        # imageio.imsave('./c3.png', data['1130'][...][..., 2])

        # NEW: aggregate guanyao's data
        data_ = []
        for key in keys:
            s_data = data[key][...]
            data_.append(s_data)
        data_ = np.array(data_)
        print('all data shape: ', data_.shape)

        # data = data['data'][:]
        # print(data)

        # print(data.shape)
        # np.random.shuffle(data)
        # data_ = []
        # grad_ = []
        # en_ = []

        np.random.shuffle(data_)
        # print(data.shape)
        self.data = np.transpose(data_, (0, 3, 2, 1))
        # Normalize to [0, 1]
        self.data = self.data / 255.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traindata = []
        grad, en = [], []
        # for i in range(self.arg.num_task):
        # traindata.append(self.source_data[i][idx])
        # grad.append(self.grad[i][idx])
        # en.append(self.en[i][idx])
        # traindata = np.array(traindata)
        # grad = np.array(grad)
        # en = np.array(en)
        # print(traindata.shape)

        # if self.transform:

        data = ((self.data[idx] - 0.5) / 0.5).astype(np.float32)
        # traindata = (traindata-0.5)/0.5
        # print(traindata)
        return data
    # return (traindata, grad, en)


from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

import torchvision.transforms as transforms


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):

        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        # resize

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        img = (img - 0.5) / 0.5
        mask = (mask - 0.5) / 0.5
        # print("---------------------------------")
        # print("MAX", img.max())
        # print("MIN", img.min())
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name': idx
        }


class MaskDataset(Dataset):
    def __init__(self, vis_dir, ir_dir, masks_dir, scale=1):
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(vis_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):

        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        # resize

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        vis_file = glob(self.vis_dir + idx + '.*')
        ir_file = glob(self.ir_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(ir_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {ir_file}'
        assert len(vis_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {vis_file}'
        mask = Image.open(mask_file[0])
        img_vis = Image.open(vis_file[0]).convert('L')
        img_ir = Image.open(ir_file[0]).convert('L')

        assert img_vis.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img_vis.size} and {mask.size}'

        img_vis = self.preprocess(img_vis, self.scale)
        mask = self.preprocess(mask, self.scale)
        img_ir = self.preprocess(img_ir, self.scale)

        img_vis = (img_vis - 0.5) / 0.5
        # mask = (mask - 0.5)/0.5
        img_ir = (img_ir - 0.5) / 0.5

        # print("---------------------------------")
        # print("MAX", img.max())
        # print("MIN", img.min())
        return {
            'vis': torch.from_numpy(img_vis).type(torch.FloatTensor),
            'ir': torch.from_numpy(img_ir).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name': idx
        }
