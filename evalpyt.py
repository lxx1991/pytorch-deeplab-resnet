import cv2
import numpy as np
import torch
from torch.autograd import Variable
import deeplab_resnet
import os

from docopt import docopt

docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage:
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapPrefix=<str>          Snapshot [default: VOC12_scenes_]
    --testGTpath=<str>          Ground truth path prefix [default: data/VOC_arg/SegmentationClass_label/]
    --testIMpath=<str>          Sketch images path prefix [default: data/VOC_arg/JPEGImages/]
    --LISTpath=<str>            Input image number list file [default: data/VOC_arg/ImageSets/Segmentation/val.txt]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --gpu0=<int>                GPU number [default: 1]
"""

args = docopt(docstr, version='v0.1')
print(args)

max_label = int(args['--NoLabels']) - 1  # labels from 0,1, ... 20(for VOC)
colors = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
colors[0, :] = 0
colors[255, :] = 255


def get_iou(pred, gt):
    if pred.shape != gt.shape:
        print('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)

    gt = gt.astype(int)
    pred = pred.astype(int)
    locs = (gt < 255)
    sumim = gt + pred * (max_label + 1)
    hs = np.bincount(sumim[locs], minlength=(max_label + 1)**2).reshape((max_label + 1), (max_label + 1))

    return hs


gpu0 = int(args['--gpu0'])
im_path = args['--testIMpath']
model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))
model.eval()
counter = 0
model.cuda(gpu0)
snapPrefix = args['--snapPrefix']
gt_path = args['--testGTpath']
img_list = open(args['--LISTpath']).readlines()

for iter in range(1, 21):  # TODO set the (different iteration)models that you want to evaluate on. Models are saved during training after each 1000 iters by default.
    saved_state_dict = torch.load(os.path.join('data/snapshots/', snapPrefix + str(iter) + '000.pth'))
    if counter == 0:
        print(snapPrefix)
    counter += 1
    model.load_state_dict(saved_state_dict)

    hist = np.zeros((max_label + 1, max_label + 1))

    for i in img_list:
        img = np.zeros((513, 513, 3))

        img_temp = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg')).astype(float)
        img_original = img_temp.copy()
        img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
        img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
        img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675

        h_offset = int((513 - img_temp.shape[0]) / 2)
        w_offset = int((513 - img_temp.shape[1]) / 2)

        img[h_offset:h_offset + img_temp.shape[0], w_offset:w_offset + img_temp.shape[1], :] = img_temp

        gt = cv2.imread(os.path.join(gt_path, i[:-1] + '.png'), cv2.IMREAD_UNCHANGED).astype(np.uint8)

        output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0, 3, 1, 2)).float(), volatile=True).cuda(gpu0))
        output = output[3].cpu().data[0].numpy()
        output = output[:, h_offset:h_offset + img_temp.shape[0], w_offset:w_offset + img_temp.shape[1]]

        output = output.transpose(1, 2, 0)
        output = np.argmax(output, axis=2).astype(np.uint8)
        if args['--visualize']:
            cv2.imshow('img', img_original.astype(np.uint8))
            gt_show = np.dstack((colors[gt, 0], colors[gt, 1], colors[gt, 2])).astype(np.uint8)
            cv2.imshow('gt', gt_show)
            output_show = np.dstack((colors[output, 0], colors[output, 1], colors[output, 2])).astype(np.uint8)
            cv2.imshow('pred', output_show)
            cv2.waitKey(10)

        hist += get_iou(output, gt)

        miou = np.diag(hist) / (1e-20 + hist.sum(1) + hist.sum(0) - np.diag(hist))
        print('pytorch', iter, "Mean iou = ", np.sum(miou) / len(miou))
