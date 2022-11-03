import os
import sys
import time
import torch
import h5py
import cv2
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import scipy.misc
from typing import List
from torch.utils.data import DataLoader
from torchvision import transforms

from train_tasks import train, finetune, train_noCrop, collab_multiNegative, finetune_multiNegative
from model import Unet_resize_conv, FusionNet, Unet_res18, Unet_train, Unet, Unet_wo_CA
from utils import YCbCr2RGB,CbCrFusion
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from torchvision import utils as vutils
argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
argparser.add_argument('--num_task', type=int, help='k shot for support set', default=3)
argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=1e-4)
argparser.add_argument('--bs', type=int, help='batch size', default=10)
argparser.add_argument('--logdir', type=str, default='logs/')
argparser.add_argument('--train', action='store_true')
argparser.add_argument('--train_collab', action='store_true')
argparser.add_argument('--train_2', action='store_true')
argparser.add_argument('--train_UNet', action='store_true')
argparser.add_argument('--test', action='store_true')
argparser.add_argument('--testVisual', action='store_true')
argparser.add_argument('--test_attention', action='store_true')
argparser.add_argument('--max_pool',action='store_true')
argparser.add_argument('--resume',action='store_true')
argparser.add_argument('--finetune',action='store_true')
argparser.add_argument('--finetune_train',action='store_true')
argparser.add_argument('--finetune_multiNegative',action='store_true')
argparser.add_argument('--pretrain', action='store_true')
argparser.add_argument('--use_gpu', action='store_true')
argparser.add_argument('--gn', action='store_true')
argparser.add_argument('--dwa', action='store_true')
argparser.add_argument('--pc', action ='store_true')
argparser.add_argument('--w', action ='store_true')
argparser.add_argument('--fs', type=int, help='fusion strategy,0~6', default=0)
argparser.add_argument('--task', type=int,help='task 0,1,2(visir,me,mf)', default=0)
argparser.add_argument('--save_dir', type=int,help='1,2,3,4,5,6', default=0)
argparser.add_argument('--c1', type=float,help='weight grad', default=0.5)
argparser.add_argument('--c2', type=float,help='weight entropy', default=0.5)
argparser.add_argument('--contrast', type=float,help='contrastive loss weight', default=1.0)
argparser.add_argument('--w_loss', type=float,help='weight of self-adaptive loss', default=1.0)

args = argparser.parse_args()

def test(model, vis_path, ir_path, f, filepath, save_path, pre, logs=None):

	if args.save_dir == 1:
		checkpath = './checkpoints_1/latest.pth'
	elif args.save_dir == 2:
		checkpath = './checkpoints_2/latest.pth'
	elif args.save_dir == 3:
		checkpath = './checkpoints_3/latest.pth'
	elif args.save_dir == 4:
		checkpath = './checkpoints_4/latest.pth'
	elif args.save_dir == 5:
		checkpath = './checkpoints_5/latest.pth'
	elif args.save_dir == 6:
		checkpath = './checkpoints_6/latest.pth'

	save_path = '/data/lrj/Template/results/' + args.logdir.split('/')[-1] + '/'
	checkpath =  './logs/latest.pth'#  './logs/latest.pth' './checkpoints_mri/latest.pth''./logs/latest_mri_7e.pth'
	print('Loading from {}...'.format(checkpath))
	
	vis_list = [n for n in os.listdir(vis_path)]
	ir_list = vis_list

	logs = torch.load(checkpath) # use checkpoints when testing
	device = torch.device('cpu')
	if args.use_gpu:
		device = torch.device('cuda')

	model.load_state_dict(logs['state_dict'])
	model.to(device)

	transform = transforms.Compose([
		transforms.ToTensor()])
	tail = vis_list[0].split('.')[-1]
	
	import time
	Time = []
	s = []
	for vis_, ir_ in zip(vis_list, ir_list):
		ir = ir_path + '/' + ir_
		vis = vis_path + '/' + vis_
		start = time.time()
		slis = False
		resi = False
		si = 2
		if f == 0:
			img1 = imageio.imread(vis).astype(np.float32)
			img2 = imageio.imread(ir).astype(np.float32)

			if resi:
				w0,h0 = img1.shape
				img1 = imageio.imresize(img1, (w0//si, h0//si)).astype(np.float32)
				img2 = imageio.imresize(img2, (w0//si, h0//si)).astype(np.float32)

			img1_data = transform(img1)  # /255. - 0.5)/0.5
			img2_data = transform(img2)

		if f == 1 or f ==2:
			img1 =imageio.imread(vis, mode='YCbCr').astype(np.float32)
			img2 =imageio.imread(ir, mode='YCbCr').astype(np.float32)
			w0,h0,c0 = img1.shape
			
			Cb1,Cr1 = img1[:,:,1], img1[:,:,2]
			Cb2,Cr2= img2[:,:,1],img2[:,:,2]
			w,h = Cb1.shape[0], Cb1.shape[1]
			
			Cb = CbCrFusion(Cb1,Cb2,w,h).reshape([w,h,1])
			Cr = CbCrFusion(Cr1,Cr2,w,h).reshape([w,h,1])
			img1_ = img1[:,:,0]/255.0
			img2_ = img2[:,:,0]/255.0

			img1_data = transform(img1_)
			img2_data = transform(img2_)
		if f ==3:
			img1 =scipy.misc.imread(vis, mode='YCbCr').astype(np.float32)
			Cb,Cr = img1[:,:,1], img1[:,:,2]
			w,h = Cb.shape[0], Cb.shape[1]
			img1_ = img1[:,:,0]/255.0

			Cb = Cb.reshape([w,h,1])
			Cr = Cr.reshape([w,h,1])
			img1_data = transform(img1_)


		img1_data = torch.unsqueeze(img1_data, 0).to(device)
		img2_data = torch.unsqueeze(img2_data, 0).to(device)

		if slis:
			s = 84
			output = img1_data
			h0, w0 = img1_data.shape[2], img1_data.shape[3]
			b_h = h0//s
			b_w = w0//s
			for i in range(b_h):
				for j in range(b_w):
					output[:,:,i*s:min((i+1)*s,h0-1),j*s:min((j+1)*s,w0-1)] = model(img1_data[:,:,i*s:min((i+1)*s,h0-1),j*s:min((j+1)*s,w0-1)], img1_data[:,:,i*s:min((i+1)*s,h0-1),j*s:min((j+1)*s,w0-1)])
		else:
			print('Image max: ', img1_data.max())
			print('Image min: ', img1_data.min())
			output = model(img1_data, img2_data)
		torch.cuda.synchronize()

		output = np.transpose((torch.squeeze(output,0).cpu().detach().numpy()*127.5+127.5), (1,2,0)).astype(np.float32)
		if f==1 or f==2:
			R,G,B = YCbCr2RGB(output,Cb,Cr)
			output = np.concatenate((B,G,R),2)

			img1 =cv2.imread(vis).astype(np.float32)
			img2 =cv2.imread(ir).astype(np.float32)
		if f==3:
			R,G,B = YCbCr2RGB(output,Cb,Cr)
			output = np.concatenate((B,G,R),2)
		# output = cv2.hconcat([img1, img2, output])
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		if resi:
			output = cv2.resize(output, (h0,w0))

		print(save_path+pre+vis_.split('.')[0]+'.bmp')
		cv2.imwrite(save_path+pre+vis_.split('.')[0]+'.bmp', output)

		end = time.time()
		Time.append(end-start)


	print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))


def main():
	print('cuda ', torch.cuda.is_available())
	print('training',args.train)

	if args.use_gpu:
		model = Unet_resize_conv().to(torch.device('cuda'))
	else:
		model = Unet_resize_conv().to(torch.device('cpu'))

	tmp = filter(lambda x: x.requires_grad, model.parameters())
	num = sum(map(lambda x: np.prod(x.shape), tmp))

	print('Total trainable tensors:', num)

	optim = torch.optim.Adam(model.parameters(), lr = args.lr)

	filepath = './logs/latest.pth'
	if args.train:
		model.train()
		data_path = []
		dir_vis = "../TNO_/vis/"  # RoadScene
		dir_ir = "../TNO_/ir/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		# crop: data1_path

		train(model, data1_path, optim, args.epoch, filepath, args)
		save_path = 'results/'+args.logdir.split('/')[-1]+'/'

	elif args.train_collab:
		model.train()
		data_path = []
		dir_vis = "../TNO_/vis/"  # RoadScene
		dir_ir = "../TNO_/ir/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		collab_multiNegative(model, data1_path, optim, args.epoch, filepath, args)
		save_path = 'results/'+args.logdir.split('/')[-1]+'/'
		test(model,'../TNO_/vis','../TNO_/ir',0, filepath, save_path, 'TNO_')
		
	elif args.train_2:
		model = Unet_train().to(torch.device('cuda'))
		optim = torch.optim.Adam(model.parameters(), lr=args.lr)
		model.train()
		data_path = []
		dir_vis = "../TNO_/vis/"  # RoadScene
		dir_ir = "../TNO_/ir/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		# crop: data1_path
		train(model, data1_path, optim, args.epoch, filepath, args)
		save_path = 'results/' + args.logdir.split('/')[-1] + '/'
		test(model, '../TNO_/vis', '../TNO_/ir', 0, filepath, save_path, 'TNO_')

	elif args.test:
		# model = Unet().to(torch.device('cuda'))
		# model = Unet_resize_conv().to(torch.device('cuda'))
		# TEST DIRECTORY
		dir_vis = "./RoadScene/vi_y" #./RoadScene/vi_y  "../CTest/vis/"  "./test/mri_ct/mri"
		dir_ir =  "./RoadScene/ir"# "../CTest/ir/" "./test/Y/petY/" "./test/mri_ct/oth/"
		# model = Unet_pretrain().to(torch.device('cuda'))
		save_path = './results/' + args.logdir.split('/')[-1] + '/'
		print('savepath: ', save_path)
		data_path = []
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		# load from ./checkpoints folder
		# logs = torch.load('./checkpoints/latest.pth')
		test(model, dir_vis, dir_ir, 0, filepath, save_path, '')

	elif args.finetune:
		model.train()
		data_path = []
		dir_vis = "../Train_vis/"  # RoadScene ./dataset/vis/
		dir_ir = "../Train_ir/"
		dir_masks = "../Train_ir_mask/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		data_path.append(dir_masks)

		# load .pth in ./logs
		finetune(model, data_path, optim, args.epoch, filepath, args)
		if args.save_dir == 0:
			save_path = 'results/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 1:
			save_path = 'results_1/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 2:
			save_path = 'results_2/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 3:
			save_path = 'results_3/' + args.logdir.split('/')[-1] + '/'
		test(model, '../TNO_/vis', '../TNO_/ir', 0, filepath, save_path, 'TNO_')
		
	elif args.finetune_multiNegative:
		model.train()
		data_path = []
		dir_vis = "../Train_vis/"  # RoadScene ./dataset/vis/
		dir_ir = "../Train_ir/"
		dir_masks = "../Train_ir_mask/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		data_path.append(dir_masks)

		# load .pth in ./logs
		finetune_multiNegative(model, data_path, optim, args.epoch, filepath, args)
		if args.save_dir == 0:
			save_path = 'results/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 1:
			save_path = 'results_1/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 2:
			save_path = 'results_2/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 3:
			save_path = 'results_3/' + args.logdir.split('/')[-1] + '/'

if __name__ == '__main__':
	main()
