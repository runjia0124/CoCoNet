import torch
from torch import nn
from torch.nn import functional as F


def fname_presuffix(fname, prefix="", suffix="", newpath=None, use_ext=True):
    """Manipulates path and name of input filename

    Parameters
    ----------
    fname : string
        A filename (may or may not include path)
    prefix : string
        Characters to prepend to the filename
    suffix : string
        Characters to append to the filename
    newpath : string
        Path to replace the path of the input fname
    use_ext : boolean
        If True (default), appends the extension of the original file
        to the output name.

    Returns
    -------
    Absolute path of the modified filename

    >>> from nipype.utils.filemanip import fname_presuffix
    >>> fname = 'foo.nii.gz'
    >>> fname_presuffix(fname,'pre','post','/tmp')
    '/tmp/prefoopost.nii.gz'

    >>> from nipype.interfaces.base import Undefined
    >>> fname_presuffix(fname, 'pre', 'post', Undefined) == \
            fname_presuffix(fname, 'pre', 'post')
    True

    """
    pth, fname, ext = split_filename(fname)
    if not use_ext:
        ext = ""

    # No need for isdefined: bool(Undefined) evaluates to False
    if newpath:
        pth = os.path.abspath(newpath)
    return os.path.join(pth, prefix + fname + suffix + ext)


def split_filename(fname):
    """Split a filename into parts: path, base filename and extension.

    Parameters
    ----------
    fname : str
        file or path name

    Returns
    -------
    pth : str
        base path from fname
    fname : str
        filename from fname, without extension
    ext : str
        file extension from fname

    Examples
    --------
    >>> from nipype.utils.filemanip import split_filename
    >>> pth, fname, ext = split_filename('/home/data/subject.nii.gz')
    >>> pth
    '/home/data'

    >>> fname
    'subject'

    >>> ext
    '.nii.gz'

    """

    special_extensions = [".nii.gz", ".tar.gz", ".niml.dset"]

    pth = os.path.dirname(fname)
    fname = os.path.basename(fname)

    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = os.path.splitext(fname)

    return pth, fname, ext


def YCbCr2RGB(Y, Cb, Cr):
    R = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
    G = 1.164 * (Y - 16) - 0.392 * (Cb - 128) - 0.813 * (Cr - 128)
    B = 1.164 * (Y - 16) + 2.017 * (Cb - 128)
    # print(R,G,B)
    return R, G, B


def CbCrFusion(Cb1, Cb2, w, h):
    Cb = Cb1
    for i in range(w):
        for j in range(h):
            if (abs(Cb1[i][j] - 128)) == 0 and (abs(Cb2[i][j]) - 128 == 0):
                Cb[i][j] = 128
            else:
                middle_1 = Cb1[i][j] * abs(Cb1[i][j] - 128) + Cb2[i][j] * abs(Cb2[i][j] - 128)
                middle_2 = abs(Cb1[i][j] - 128) + abs(Cb2[i][j] - 128)
                # print(middle_1,middle_2)
                Cb[i][j] = middle_1 / middle_2
    return Cb


class Gradient_Net_iqa(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(Gradient_Net_iqa, self).__init__()
        # kernel_x = [[1/8, 1/8, 1/8], [1/8, -1, 1/8], [1/8., 1/8, 1/8]]
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_x = kernel_x.expand(1, 1, 3, 3)  # .to(device)
        # kernel_x = torch.stack([kernel_x,kernel_x,kernel_x],0)
        # kernel_x = torch.stack([kernel_x,kernel_x,kernel_x],0).to(device)
        # kernel_y = [[1/8, 1/8, 1/8], [1/8, -1, 1/8], [1/8., 1/8, 1/8]]
        # kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        # kernel_y = kernel_y.expand(1, 1, 3, 3)#.to(device)
        # kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.expand(1, 1, 3, 3)  # .to(device)
        # kernel_y = torch.stack([kernel_y,kernel_y,kernel_y],0)
        # kernel_y = torch.stack([kernel_y,kernel_y,kernel_y],0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        # print('ca')
        # gradient = x
        # n,c,w,h = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        # for i in range(n):
        # 	for j in range(c):
        # 		y = x[i][j]
        # 		y = y.view([1,1,w,h])
        # 		grad_x = F.conv2d(y, self.weight_x, stride=1, padding=1)
        # 		grad_y = F.conv2d(y, self.weight_y, stride=1, padding=1)
        # 		gradient[i][j] = torch.abs(grad_x) + torch.abs(grad_y)
        # for i in range(c):
        # 	y = x[:,i]
        # 	y = y.view([n,1,w,h])
        grad_x = F.conv2d(x, self.weight_x, stride=1, padding=1)
        grad_y = F.conv2d(x, self.weight_y, stride=1, padding=1)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        # print(gradient.shape)
        # gradient = torch.abs(gradient)
        return gradient + 0.00001


class Gradient_Net(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(Gradient_Net, self).__init__()
        kernel_x = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8., 1 / 8, 1 / 8]]
        # kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_x = kernel_x.expand(1, 1, 3, 3)
        # kernel_x = torch.stack([kernel_x,kernel_x,kernel_x],0)
        # kernel_x = torch.stack([kernel_x,kernel_x,kernel_x],0).to(device)
        kernel_y = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8., 1 / 8, 1 / 8]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.expand(1, 1, 3, 3)  # .to(device)
        # kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        # kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        # kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        # kernel_y = kernel_y.expand(1, 1, 3, 3)#.to(device)
        # kernel_y = torch.stack([kernel_y,kernel_y,kernel_y],0)
        # kernel_y = torch.stack([kernel_y,kernel_y,kernel_y],0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        gradient = x
        n, c, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        for i in range(n):
            for j in range(c):
                y = x[i][j]
                y = y.view([1, 1, w, h])
                grad_x = F.conv2d(y, self.weight_x, stride=1, padding=1)
                # grad_y = F.conv2d(y, self.weight_y, stride=1, padding=1)
                gradient[i][j] = torch.abs(grad_x)  # + torch.abs(grad_y)
        # for i in range(c):
        # 	y = x[:,i]
        # 	y = y.view([n,1,w,h])
        # 	grad_x = F.conv2d(y, self.weight_x, stride=1, padding=1)
        # 	grad_y = F.conv2d(y, self.weight_y, stride=1, padding=1)
        # 	gradient[:,i] = torch.abs(grad_x) + torch.abs(grad_y)
        # print(gradient.shape)
        # gradient = torch.abs(gradient)
        # print(gradient.shape,grad_x.shape)
        return gradient + 0.00001


def gradient(x, gradient_model):
    g = gradient_model(x)
    return g


class Mean_Net(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(Mean_Net, self).__init__()
        # kernel_x = [[1/8, 1/8, 1/8], [1/8, -1, 1/8], [1/8., 1/8, 1/8]]
        kernel = [[1 / 4, 1 / 4, 1 / 4], [1 / 4, 1, 1 / 4], [1 / 4, 1 / 4, 1 / 4]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(1, 1, 3, 3)  # .to(device)

        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        mean = x
        x = torch.abs(x)
        n, c, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        for i in range(n):
            for j in range(c):
                y = x[i][j]
                y = y.view([1, 1, w, h])
                mean[i][j] = F.conv2d(y, self.weight, stride=1, padding=1)
        # mean = torch.abs(mean)
        # gradient = torch.abs(gradient)
        return mean


def mean(x, mean_model):
    m = mean_model(x)
    return m


from torch.optim import lr_scheduler
from PIL import Image
import collections
import torch.nn.init as init
import numpy as np
import inspect, re
import os
import math


def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


def atten2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor[0]
    image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), 0)
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy / (image_numpy.max() / 255.0)
    return image_numpy.astype(imtype)


def latent2im(image_tensor, imtype=np.uint8):
    # image_tensor = (image_tensor - torch.min(image_tensor))/(torch.max(image_tensor)-torch.min(image_tensor))
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


def max2im(image_1, image_2, imtype=np.uint8):
    image_1 = image_1[0].cpu().float().numpy()
    image_2 = image_2[0].cpu().float().numpy()
    image_1 = (np.transpose(image_1, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_2 = (np.transpose(image_2, (1, 2, 0))) * 255.0
    output = np.maximum(image_1, image_2)
    output = np.maximum(output, 0)
    output = np.minimum(output, 255)
    return output.astype(imtype)


def variable2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].data.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir,
                                                                                                        'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)

    return init_fun
