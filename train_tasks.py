import visdom
from pytorch_ssim import ssim
from visualizer import Visualizer
from torch.utils.data import DataLoader

from dataset import TrainDataSet as trainset
from utils import *
from save_image import normimage
from checkpoint import save_latest, save_latest_finetune
from P_loss import ContrastiveLoss_multiNegative


device = torch.device('cuda')


def train(model, data_path, optimizer, args):

	checkpath = args.logdir
	if not os.path.exists(checkpath):
		os.makedirs(checkpath)
	
	mo = 'w'
	if args.resume:
		mo = 'a'

	f = open(args.logdir + '/log.txt', mo)
	print('total e:', args.epoch)
	e = 0
	if args.resume:
		logs = torch.load(args.resume_ckpt)
		model.load_state_dict(logs['model_state_dict'])
		e = logs['epoch']

	print("-----------------Parameters--------------------")
	print('Epoch: ', args.epoch)
	print('Dataset: ', data_path)
	print('Checkpoint save path: ', checkpath)
	print(args.resume, ' Resume from: ', args.resume_ckpt)
	print('C param: {}, {}'.format(args.c1, args.c2))
	print("-----------------------------------------------")
	f.write('-----------------Parameters--------------------'+'\n')
	f.write('Epoch: {}'.format(args.epoch) + '\n')
	f.write('Dataset: {}'.format(data_path) + '\n')
	f.write('Ckpt path: {}'.format(checkpath)  + '\n')
	f.write('If True, resume from: {}'.format(args.resume_ckpt) + '\n')
	f.write('C param: {}, {}'.format(args.c1, args.c2) + '\n')
	f.write('-----------------------------------------------' + '\n')
	
	visualizer = Visualizer()
	visd = visdom.Visdom()
	ite_num = 0

	model = model.to(device)
	for i in range(e, args.epoch):
		data_loader = DataLoader(
			trainset(data_path, arg=args),
			batch_size=args.bs,
			shuffle=True
		)
		train_ep(
			epoch_idx=i, 
			model=model, 
			data_loader=data_loader, 
			ite_num=ite_num, 
			optimizer=optimizer, 
			visualizer=visualizer, 
			visd=visd, 
			f=f, args=args
		)

	f.close()


def train_ep(epoch_idx, model, data_loader, ite_num, optimizer, visualizer, visd, f, args):

	t = enumerate(iter(data_loader))
	for batch_idx, batch in t:

		ite_num = ite_num + 1

		x1 = batch[:, 0, :, :]
		x2 = batch[:, 1, :, :]

		n, w, h = x1.shape[0], x1.shape[1], x1.shape[2]
		x1 = x1.view([n, 1, w, h]).to(device)
		x2 = x2.view([n, 1, w, h]).to(device)

		# calculate self-adaptive weights
		weights = measure_module1(x1, x2, args)  
		y = model(x1, x2)

		# calculate total loss
		loss = loss_fc(args, x1, x2, y, weights)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses = collections.OrderedDict()
		losses['loss'] = loss.data.cpu()
		losses[' '] = 0
		visualizer.plot_current_losses(ite_num + 1,
										float(epoch_idx) / len(data_loader),
										losses)

		if ite_num % 5 == 0:
			visshow = normimage(x1, save_cfg=False)
			irshow = normimage(x2, save_cfg=False)
			outshow = normimage(y, save_cfg=False)
			shows = []
			shows.append(visshow)
			shows.append(irshow)
			shows.append(outshow)
			visd.images(shows, nrow=4, padding=3, win=1, opts=dict(title='Output images'))
			# save checkpoint
			save_latest(model, optimizer, args.logdir, epoch_idx, ite_num)

		if batch_idx % 10 == 0 and batch_idx != 0:
			print('epoch: {}, batch: {}, total_loss: {}'.format(epoch_idx, batch_idx, loss))
		
		f.write(str(loss.cpu().detach().numpy())+'\n')


def finetune_multiNegative(model, data_path, optimizer, args):
	"""
	Finetune with contrastive loss, multiple negative samples
	"""
	epoch = args.epoch
	checkpath = args.logdir
	filepath = os.path.join(checkpath, 'latest.pth')

	f = open(args.logdir + '/log.txt', 'a')
	f.write('-------------Finetune Parameters----------------'+'\n')
	f.write('Epoch: {}'.format(epoch) + '\n')
	f.write('Dataset: {}'.format(data_path) + '\n')
	f.write('Ckpt path: {}'.format(checkpath)  + '\n')
	f.write('Finetune from: {}'.format(filepath) + '\n')
	f.write('C param: {}, {}'.format(args.c1, args.c2) + '\n')
	f.write('-----------------------------------------------' + '\n')

	print(f'Loading from {filepath} to finetune...')
	logs = torch.load(filepath)

	device = torch.device('cpu')
	if args.use_gpu:
		device = torch.device('cuda')
	model.load_state_dict(logs['state_dict'])
	optimizer.load_state_dict(logs['optimizer'])
	model.to(device)

	if not os.path.exists(checkpath):
		os.makedirs(checkpath)
	
	print('total e:', epoch)
	e = 0
	if args.resume:
		logs = torch.load(filepath)
		model.load_state_dict(logs['model_state_dict'])
		e = logs['epoch']

	visualizer = Visualizer()
	visd = visdom.Visdom()
	ite_num = 0

	model = model.to(device)
	
	for i in range(e, epoch):
		data_loader = DataLoader(
			trainset(data_path, arg=args),
			batch_size=args.bs,
			shuffle=True,
			drop_last=True
		)
		finetune_ep(
			epoch_idx=i, 
			model=model, 
			data_loader=data_loader, 
			ite_num=ite_num, 
			optimizer=optimizer, 
			visualizer=visualizer, 
			visd=visd, 
			f=f, args=args
		)

	f.close()


def finetune_ep(epoch_idx, model, data_loader, ite_num, optimizer, visualizer, visd, f, args):
	"""Finetuning with contrastive loss
	"""
	t = enumerate(iter(data_loader))
	for batch_idx, batch in t:

		ite_num = ite_num + 1

		x1 = batch[:, 0, :, :].to(device)
		x2 = batch[:, 1, :, :].to(device)
		mask = batch[:, 2, :, :].to(device)
		mask = mask * 0.5 + 0.5

		n, w, h = x1.shape[0], x1.shape[1], x1.shape[2]
		x1 = x1.view([n, 1, w, h])
		x2 = x2.view([n, 1, w, h])
		mask = mask.view([n, 1, w, h])

		x1 = x1.to(device)
		x2 = x2.to(device)
		mask = mask.to(device)

		# load three groups of image pair,
		# one pair as anchor and positive sample,
		# the other two pair as anchor and negative samples
		vis_1 = x1[0:10, :, :,:]
		vis_2 = x1[10:20, :, :,:]
		vis_3 = x1[20:30, :, :,:]
		ir_1 = x2[0:10, :, :,:]
		ir_2 = x2[10:20, :, :,:]
		ir_3 = x2[20:30, :, :,:]
		
		# load mask, skip zero mask patch to avoid gradient explosion
		mask_1 = mask[0:10, :, :, :]
		if mask_1.max() == 0:
			continue
		
		# calculate adaptive weights
		weights = measure_module1(vis_1, ir_1, args)  # c[args.task]
		y = model(vis_1, ir_1)

		# for contrastive loss
		positive_ir = ir_1 * mask_1
		negative_ir_1 = vis_1 * mask_1
		negative_ir_2 = vis_2 * mask_1
		negative_ir_3 = vis_3 * mask_1

		positive_vis = vis_1 * (1 - mask_1)
		negative_vis_1 = ir_1 * (1 - mask_1)
		negative_vis_2 = ir_2 * (1 - mask_1)
		negative_vis_3 = ir_3 * (1 - mask_1)

		# contrastive loss with mask
		vgg19_weights = 'placeholder'
		c_loss = ContrastiveLoss_multiNegative(vgg19_weights).cuda()
		contrast_loss = c_loss(negative_ir_1, negative_ir_2, negative_ir_3, y * mask_1, positive_ir) + \
						c_loss(negative_vis_1, negative_vis_2, negative_vis_3, y * (1 - mask_1), positive_vis)
		contrast_loss = contrast_loss.cuda()

		# total loss
		loss = args.contrast * contrast_loss + loss_fc(args, vis_1, ir_1, y, weights)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses = collections.OrderedDict()
		losses['loss'] = loss.data.cpu()
		losses['contrast_loss'] = contrast_loss.data.cpu()
		visualizer.plot_current_losses(ite_num + 1,
										float(epoch_idx) / len(data_loader),
										losses)

		visshow = normimage(x1, save_cfg=False)
		irshow = normimage(x2, save_cfg=False)
		outshow = normimage(y, save_cfg=False)
		pos_irshow = normimage(positive_ir, save_cfg=False)
		pos_visshow = normimage(positive_vis, save_cfg=False)

		shows = []
		shows.append(visshow)
		shows.append(irshow)
		shows.append(outshow)
		shows.append(pos_irshow)
		shows.append(pos_visshow)
		visd.images(shows, nrow=4, padding=3, win=1, opts=dict(title='Output images'))

		if ite_num % 5 == 0:
			save_latest_finetune(model, optimizer, args.logdir, epoch_idx, ite_num)

		if batch_idx % 100 == 0 and batch_idx != 0:
			print('epoch: {}, batch: {}, total_loss: {}'.format(epoch_idx, batch_idx, loss))
		
		f.write(str(loss.cpu().detach().numpy())+'\n')


def softmax(x, T):
	x = [np.exp(i.cpu().detach()/T) for i in x]
	sum_ = sum(x)
	x = [len(x)*i/sum_ for i in x]
	return	x


def measure_module1(x1, x2, args):
	"""
	Measure the gradient and entropy,
	and calculate weights
	"""
	c1 = args.c1
	c2 = args.c2
	gm1, m1 = measure_info(x1)
	gm2, m2 = measure_info(x2)

	output = []
	with torch.no_grad():
		m = nn.Softmax(dim=0)
		sum_ = gm1 + gm2
		e = torch.stack((gm1/c1/sum_,gm2/c1/sum_),0)
		output.append(m(e))

		sum_ = m1 + m2
		e = torch.stack((m1/c2/sum_,m2/c2/sum_),0)
		output.append(m(e))
	return output


def measure_info(x):
	"""
	calculate mean gradient and entropy
	"""
	with torch.no_grad():
		grad_model = Gradient_Net_iqa().to(x.device)
		grad = gradient(x, grad_model)

		grad_mean = grad.mean(dim=(1,2,3))
		en = entropy(x)
		en = torch.from_numpy(en).cuda()
		return grad_mean, en


def entropy(x):

	len = x.shape[0]
	entropies = np.zeros(shape = (len))
	grey_level = 256
	counter = np.zeros(shape = (grey_level, 1))

	for i in range(len):
		input_uint8 = (x[i, 0, :, :] * 127.5 + 127.5).cpu().detach().numpy().astype(np.uint8)
		W = x.shape[2]
		H = x.shape[3]
		for m in range(W):
			for n in range(H):
				indexx = input_uint8[m, n]
				counter[indexx] = counter[indexx] + 1
		total = np.sum(counter)
		p = counter / total
		for k in range(grey_level):
			if p[k] != 0:
				entropies[i] = entropies[i] - p[k] * np.log2(p[k])
				
		entropies += 0.0001
	return entropies


def loss_fc(args, x1,x2,y,w=[[0.5,0.5],[0.5,0.5]]):

	MSEloss = nn.MSELoss()

	len_ = x1.shape[0]
	mse_loss = []
	for i in range(len_):
		loss = w[1][0][i]*MSEloss(y[i], x1[i]) + w[1][1][i]*MSEloss(y[i], x2[i])
		mse_loss.append(loss)
	mse_loss = torch.stack(mse_loss)
	mse_loss = torch.mean(mse_loss)

	x1 = x1 * 0.5 + 0.5
	x2 = x2 * 0.5 + 0.5
	y = y * 0.5 + 0.5
	ssim_1 = ssim(x1,y)
	ssim_2 = ssim(x2,y)

	ssim_loss = w[0][0]*(1 - ssim_1)+\
				w[0][1]*(1 - ssim_2)
	ssim_loss = torch.mean(ssim_loss)

	return 20*ssim_loss + mse_loss

