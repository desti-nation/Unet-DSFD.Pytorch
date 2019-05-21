#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from models.DSFD_Unet import DSFD_Unet
from models.factory import build_net
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr
import shutil

parser = argparse.ArgumentParser(description='dsfd demo')
# parser.add_argument('--network',
#                     default='vgg', type=str,
#                     choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
#                     help='model for training')
parser.add_argument('--save_dir',
                    type=str, default='./train/Lime_results',
                    help='Directory for detect result')
parser.add_argument('--model',
                    type=str,
                    default='weights/vgg/dsfd.pth', help='trained model')
parser.add_argument('--thresh',
                    default=0.1, type=float,
                    help='Final confidence threshold')
parser.add_argument('--img_path',
                    type=str, default='train/Lime_img/test',
                    help='Directory for detect result')
parser.add_argument('--gt_dir',
                    type=str, default='/home/lb/data/dark_face/DarkFace_Train_with_bug_new_version_being_uploaded/label',
                    help='trained model')
parser.add_argument('--dsfd_weights',
                    type=str, default=None,
                    help='trained model')
parser.add_argument('--unet_weights',
                    type=str, default=None,
                    help='trained model')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def checkpath(path):
    if not os.path.isdir(path): os.makedirs(path)


def load_gt(gt_path):
    file = open(gt_path, "r")
    data = file.read().splitlines()[1:]
    gt = []
    for t in data:
        t = t.strip().split(" ")
        t = list(map(int, t))
        gt.append(t)
    file.close()
    return gt

def detect(net, img_path, thresh, gt_path):
    img_name = img_path.split('/')[-1].split('.')[0]
    out_f = "{}/txt/{}.txt".format(args.save_dir, img_name)
    # if os.path.isfile(out_f):
    #     print("exists.")
    #     return
    fout = open(out_f, "w")

    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1500 * 1000 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # plot ground truth
    gt = load_gt(gt_path)
    for box in gt:
        left_up = (box[0], box[1])
        right_bottom = (box[2], box[3])
        cv2.rectangle(img, left_up, right_bottom, (0, 255, 78), 1)

    for i in range(detections.size(1)):
        j = 0
        while j < detections.size(2) and detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 1)
            conf = "{:.2f}".format(score)
            text_size, baseline = cv2.getTextSize(
                conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            fout.write('{} {} {} {} {}\n'.format(pt[0], pt[1], pt[2], pt[3], score))
            cv2.rectangle(img, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                          (p1[0] + text_size[0], p1[1] + text_size[1]),[255,0,0], -1)
            cv2.putText(img, conf, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)

    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))

    cv2.imwrite(os.path.join(args.save_dir, "png", os.path.basename(img_path)), img)
    fout.close()


if __name__ == '__main__':
    net = DSFD_Unet(args.dsfd_weights, args.unet_weights, mode="test")
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    checkpath(args.save_dir)
    checkpath(args.save_dir + "/png")
    checkpath(args.save_dir + "/txt")
    checkpath(args.save_dir + "/weights")
    # copy weights to save_dir
    shutil.copy(args.model, "{}/weights/weights.pth".format(args.save_dir))

    img_path = args.img_path
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('png')]
    for path in img_list:
        img_name = path.split("/")[-1].split(".")[0].split("_")[0]
        gt_path = "{}/{}.txt".format(args.gt_dir, img_name)
        detect(net, path, args.thresh, gt_path)