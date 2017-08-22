import os
import numpy as np
import cv2
from torch.utils.data import Dataset


class SegTransform(object):
    def __init__(self, output_size, rescale=(0.5, 1.5), mean_color=[104.008, 116.669, 122.675], ignore_label=255):
        self.output_size = output_size
        self.rescale = rescale
        self.mean_color = mean_color
        self.ignore_label = ignore_label

    def __call__(self, image, label):

        np.random.seed()
        scale = np.random.uniform(*self.rescale)  # random.uniform(0.5,1.5) does not fit in a Titan X with the present version of pytorch, so we random scaling in the range (0.5,1.3), different than caffe implementation in that caffe used only 4 fixed scales. Refer to read me

        if np.random.randint(0, 2) == 1:
            image = np.fliplr(image)
            label = np.fliplr(label)

        new_dims = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_dims)
        # cv2.imshow('img0', image.astype(np.uint8));

        label = cv2.resize(label, new_dims, interpolation=cv2.INTER_NEAREST)

        pad_h = max(self.output_size - image.shape[0], 0)
        pad_w = max(self.output_size - image.shape[1], 0)

        # padding
        if (pad_h > 0 or pad_w > 0):
            temp_h = np.random.randint(0, pad_h + 1)
            temp_w = np.random.randint(0, pad_w + 1)
            image = cv2.copyMakeBorder(image, temp_h, pad_h - temp_h, temp_w, pad_w - temp_w, cv2.BORDER_CONSTANT, value=self.mean_color)
            label = cv2.copyMakeBorder(label, temp_h, pad_h - temp_h, temp_w, pad_w - temp_w, cv2.BORDER_CONSTANT, value=self.ignore_label)

        # cropping
        temp_h = np.random.randint(0, image.shape[0] - self.output_size + 1)
        temp_w = np.random.randint(0, image.shape[1] - self.output_size + 1)
        image = image[temp_h:temp_h + self.output_size, temp_w:temp_w + self.output_size, :]
        label = label[temp_h:temp_h + self.output_size, temp_w:temp_w + self.output_size]

        # cv2.imshow('img', image.astype(np.uint8));
        # cv2.imshow('label', label);
        # cv2.waitKey();

        image[:, :, 0] = image[:, :, 0] - self.mean_color[0]
        image[:, :, 1] = image[:, :, 1] - self.mean_color[1]
        image[:, :, 2] = image[:, :, 2] - self.mean_color[2]

        return image, label


class VOCArgDataset(Dataset):
    def __init__(self, list_path, img_path, gt_path, transform=None):
        with open(list_path) as f:
            self.img_list = []
            for line in f:
                self.img_list.append(line[:-1])
        self.img_path = img_path
        self.gt_path = gt_path
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx] + '.jpg')).astype(np.float32)
        label = cv2.imread(os.path.join(self.gt_path, self.img_list[idx] + '.png'), cv2.IMREAD_UNCHANGED)
        if self.transform:
            image, label = self.transform(image, label)
        image = image.transpose((2, 0, 1))

        return image, label
