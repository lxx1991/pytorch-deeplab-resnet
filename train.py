from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import deeplab_resnet
import voc_arg
import cv2
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
from docopt import docopt
import time
docstr = """Train ResNet-DeepLab on VOC12 (scenes) in pytorch using MSCOCO pretrained initialization 

Usage: 
    train.py [options]

Options:
    -h, --help                  Print this message
    --GTpath=<str>              Ground truth path prefix [default: data/VOC_arg/SegmentationClass_label/]
    --IMpath=<str>              Sketch images path prefix [default: data/VOC_arg/JPEGImages/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --LISTpath=<str>            Input image number list file [default: data/VOC_arg/ImageSets/Segmentation/train.txt]
    --lr=<float>                Learning Rate [default: 0.00025]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 1]
    --wtDecay=<float>          Weight decay during training [default: 0.0005]
    --gpu0=<int>                GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
    --Ignore=<int>              Index of the ignore label [default: 255]
"""

#    -b, --batchSize=<int>       num sample per batch [default: 1] currently only batch size of 1 is implemented, arbitrary batch size to be implemented soon
args = docopt(docstr, version='v0.1')
print(args)

cudnn.enabled = False
gpu0 = int(args['--gpu0'])


def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    j = (j + 1) / 2
    j = int(np.ceil((j + 1) / 2.0))
    j = (j + 1) / 2
    return j


# def read_file(path_to_file):
#     with open(path_to_file) as f:
#         img_list = []
#         for line in f:
#             img_list.append(line[:-1])
#     return img_list

# def chunker(seq, size):
#     return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

# def scale_im(img_temp, scale):
#     new_dims = (int(img_temp.shape[0] * scale), int(img_temp.shape[1] * scale))
#     return cv2.resize(img_temp, new_dims).astype(float)

# def scale_gt(img_temp, scale):
#     new_dims = (int(img_temp.shape[0] * scale), int(img_temp.shape[1] * scale))
#     return cv2.resize(img_temp, new_dims, interpolation=cv2.INTER_NEAREST).astype(float)

# def get_data_from_chunk_v2(chunk):
#     gt_path = args['--GTpath']
#     img_path = args['--IMpath']
#     ignore_label = int(args['--Ignore'])
#     crop_dim = 321

#     images = np.zeros((crop_dim, crop_dim, 3, len(chunk)))
#     gt = np.zeros((outS(crop_dim), outS(crop_dim), len(chunk)))

#     for i, piece in enumerate(chunk):
#         scale = np.random.uniform(0.5, 1.5)  #random.uniform(0.5,1.5) does not fit in a Titan X with the present version of pytorch, so we random scaling in the range (0.5,1.3), different than caffe implementation in that caffe used only 4 fixed scales. Refer to read me
#         flip_p = np.random.randint(0, 2)
#         img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg')).astype(float)
#         gt_temp = cv2.imread(os.path.join(gt_path, piece + '.png'))[:, :, 0]
#         img_temp = scale_im(img_temp, scale)
#         gt_temp = scale_gt(gt_temp, scale)
#         pad_h = max(crop_dim - img_temp.shape[0], 0)
#         pad_w = max(crop_dim - img_temp.shape[1], 0)

#         # padding
#         if (pad_h > 0 or pad_w > 0):
#             temp_h = np.random.randint(0, pad_h + 1)
#             temp_w = np.random.randint(0, pad_w + 1)
#             img_temp = cv2.copyMakeBorder(img_temp, temp_h, pad_h - temp_h, temp_w, pad_w - temp_w, cv2.BORDER_CONSTANT, value=[104.008, 116.669, 122.675])
#             gt_temp = cv2.copyMakeBorder(gt_temp, temp_h, pad_h - temp_h, temp_w, pad_w - temp_w, cv2.BORDER_CONSTANT, value=ignore_label)

#         # cropping
#         temp_h = np.random.randint(0, img_temp.shape[0] - crop_dim + 1)
#         temp_w = np.random.randint(0, img_temp.shape[1] - crop_dim + 1)
#         img_temp = img_temp[temp_h:temp_h + crop_dim, temp_w:temp_w + crop_dim, :]
#         gt_temp = gt_temp[temp_h:temp_h + crop_dim, temp_w:temp_w + crop_dim]

#         img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
#         img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
#         img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675

#         if flip_p == 1:
#             img_temp = np.fliplr(img_temp)
#             gt_temp = np.fliplr(gt_temp)

#         images[:, :, :, i] = img_temp
#         gt[:, :, i] = cv2.resize(gt_temp, (outS(crop_dim), outS(crop_dim)), interpolation=cv2.INTER_NEAREST).astype(float)

#     images = images.transpose((3, 2, 0, 1))
#     images = torch.from_numpy(images).float()
#     gt = gt.transpose((2, 0, 1))
#     gt = torch.from_numpy(gt).long()

#     return images, gt


def loss_calc(out, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape batch_size x h x w  -> batch_size x h x w

    ignore_label = int(args['--Ignore'])
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d(ignore_index=ignore_label)
    out = m(out)

    return criterion(out, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


if not os.path.exists('data/snapshots'):
    os.makedirs('data/snapshots')

model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))

saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth')
if int(args['--NoLabels']) != 21:
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        if i_parts[1] == 'layer5':
            saved_state_dict[i] = model.state_dict()[i]

model.load_state_dict(saved_state_dict)

max_iter = int(args['--maxIter'])
batch_size = 16
weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])

model.float()
model.eval()  # use_global_stats = True

net = torch.nn.DataParallel(model.cuda(), device_ids=range(8))
criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr}, {'params': get_10x_lr_params(model), 'lr': 10 * base_lr}], lr=base_lr, momentum=0.9, weight_decay=weight_decay)

optimizer.zero_grad()

voc_arg_dataset = voc_arg.VOCArgDataset(args['--LISTpath'], args['--IMpath'], args['--GTpath'], transform=voc_arg.SegTransform(321, ignore_label=int(args['--Ignore'])))
trainloader = DataLoader(voc_arg_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
trainiter = iter(trainloader)

start = time.time()
for iter in range(max_iter + 1):

    iter_size = int(args['--iterSize'])
    tot_loss = 0
    for temp in range(iter_size):
        image, label = trainiter.next()
        image = Variable(image.float()).cuda(gpu0)
        label = Variable(label.long()).cuda(gpu0)
        out = net(image)
        loss = loss_calc(out[0], label)

        for i in range(len(out) - 1):
            loss = loss + loss_calc(out[i + 1], label)
        loss = loss / iter_size
        loss.backward()

        tot_loss += loss.data.cpu().numpy()[0]

    optimizer.step()
    lr_ = lr_poly(base_lr, iter, max_iter, 0.9)
    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_}, {'params': get_10x_lr_params(model), 'lr': 10 * lr_}], lr=lr_, momentum=0.9, weight_decay=weight_decay)
    optimizer.zero_grad()

    if iter % 10 == 0 and iter != 0:
        print('iter = ', iter, 'of', max_iter, 'completed, loss = ', tot_loss)
        print('(poly lr policy) learning rate', lr_)
        time_left = int((time.time() - start) / iter * (max_iter - iter))
        print('Time left %d:%d' % (int(time_left / 3600), int((time_left % 3600) / 60)))
    if iter % 1000 == 0 and iter != 0:
        print('taking snapshot ...')
        torch.save(model.state_dict(), 'data/snapshots/VOC12_scenes_' + str(iter) + '.pth')
end = time.time()
print(end - start, 'seconds')
