#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import preprocess


class DARKDetection(data.Dataset):
    """docstring for DARKDetection"""

    def __init__(self, list_file, mode = 'train',
                 img_path = "/home/lb/data/dark_face/DarkFace_Train_with_bug_new_version_being_uploaded/images",
                 gt_path = "/home/lb/data/dark_face/DarkFace_Train_with_bug_new_version_being_uploaded/label"):
        super(DARKDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for img_name in lines:
            img_name = img_name.strip()
            if len(img_name) == 0:
                continue
            gt_name = img_name.replace("png", "txt")
            img = os.path.join(img_path, img_name)
            gts = os.path.join(gt_path, gt_name)
            with open(gts) as f:
                gts = f.readlines()
            # gts = gts.strip().split()
            face_num = int(gts[0])
            box = []
            label = []
            for i in range(1, face_num + 1):
                x1, y1, x2, y2 = gts[i].strip().split()
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                box.append([x1, y1, x2, y2])
                label.append(1)

            if len(box) > 0:
                self.fnames.append(img)
                self.boxes.append(box)
                self.labels.append(label)


        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, h, w = self.pull_item(index)
        return img, target

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess(
                img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break 
            else:
                index = random.randrange(0, self.num_samples)

        
        #img = Image.fromarray(img)
        '''
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in sample_labels:
            bbox = (bbox[1:] * np.array([w, h, w, h])).tolist()

            draw.rectangle(bbox,outline='red')
        img.save('image.jpg')
        '''
        return torch.from_numpy(img), target, im_height, im_width
        

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    from config import cfg
    dataset = DARKDetection("/home/lb/lb/DSFD.Unet.pytorch/train_list.txt", mode='train')
    #for i in range(len(dataset)):
    dataset.pull_item(14)
