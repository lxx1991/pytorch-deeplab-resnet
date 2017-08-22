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
    --wtDecay=<float>           Weight decay during training [default: 0.0005]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
    --Ignore=<int>              Index of the ignore label [default: 255]
"""

#    -b, --batchSize=<int>       num sample per batch [default: 1] currently only batch size of 1 is implemented, arbitrary batch size to be implemented soon
args = docopt(docstr, version='v0.1')
print(args)

# cudnn.enabled = False


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
optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(net.module), 'lr': base_lr}, {'params': get_10x_lr_params(net.module), 'lr': 10 * base_lr}], lr=base_lr, momentum=0.9, weight_decay=weight_decay)

optimizer.zero_grad()

voc_arg_dataset = voc_arg.VOCArgDataset(args['--LISTpath'], args['--IMpath'], args['--GTpath'], transform=voc_arg.SegTransform(321, ignore_label=int(args['--Ignore'])))

trainloader = DataLoader(voc_arg_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
trainiter = iter(trainloader)

start = time.time()
for i in range(max_iter + 1):
    iter_size = int(args['--iterSize'])
    tot_loss = 0
    for temp in range(iter_size):
        try:
            image, label = trainiter.next()
        except StopIteration:
            trainiter = iter(trainloader)
            image, label = trainiter.next()
        image = Variable(image.float()).cuda()
        label = Variable(label.long()).cuda()
        out = net(image)

        loss = loss_calc(out[0], label)
        for j in range(len(out) - 1):
            loss = loss + loss_calc(out[j + 1], label)
        loss = loss / iter_size
        loss.backward()

        tot_loss += loss.data.cpu().numpy()[0]

    optimizer.step()
    lr_ = lr_poly(base_lr, i, max_iter, 0.9)
    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(net.module), 'lr': lr_}, {'params': get_10x_lr_params(net.module), 'lr': 10 * lr_}], lr=lr_, momentum=0.9, weight_decay=weight_decay)
    optimizer.zero_grad()

    if i % 10 == 0 and i != 0:
        print('iter = ', i, 'of', max_iter, 'completed, loss = ', tot_loss)
        print('(poly lr policy) learning rate', lr_)
        time_left = int((time.time() - start) / i * (max_iter - i))
        print('Time left %d:%d' % (int(time_left / 3600), int((time_left % 3600) / 60)))
    if i % 1000 == 0 and i != 0:
        print('taking snapshot ...')
        torch.save(model.state_dict(), 'data/snapshots/VOC12_scenes_' + str(i) + '.pth')

end = time.time()
print(end - start, 'seconds')
