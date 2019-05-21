#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import sys
sys.path.insert(0, "/home/lb/lb")
sys.path.insert(1, "/home/lb/lb/DSFD.pytorch")
from tools import *
import glob
import os
import torch
import argparse
import torch.backends.cudnn as cudnn

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from models.factory import build_net
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='dsfd evaluatuon pascal')
parser.add_argument('--network',
                    default='vgg', type=str,
                    choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--model',
                    type=str,
                    default='weights/dsfd_vgg_0.880.pth', help='trained model')
parser.add_argument('--input_path',
                    type=str,
                    default='input', help='input path')
parser.add_argument('--output_path',
                    type=str,
                    default='output', help='output_path')
parser.add_argument('--thresh', default=0.1, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

PASCAL_IMG_DIR = os.path.join(cfg.FACE.PASCAL_DIR, 'images')
PASCAL_RESULT_DIR = os.path.join(cfg.FACE.PASCAL_DIR, 'dsfd')
PASCAL_RESULT_IMG_DIR = os.path.join(PASCAL_RESULT_DIR, 'images')

if not os.path.exists(PASCAL_RESULT_IMG_DIR):
    os.makedirs(PASCAL_RESULT_IMG_DIR)


def detect_face(net, img, thresh):
    height, width, _ = img.shape
    im_shrink = 640.0 / max(height, width)
    image = cv2.resize(img, None, None, fx=im_shrink,
                       fy=im_shrink, interpolation=cv2.INTER_LINEAR).copy()

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    bboxes = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            box = []
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(np.int)
            j += 1
            box += [pt[0], pt[1], pt[2], pt[3], score]
            box[1] += 0.2 * (box[3] - box[1] + 1)
            bboxes += [box]

    return bboxes


if __name__ == '__main__':
    print("Start test on DARK FACE dataset.")
    net = build_net('test', cfg.NUM_CLASSES, args.network)
    net.load_state_dict(torch.load(args.model))
    net.eval()
    print("Load model Finished.")

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    checkpath(args.output_path)
    checkpath(args.output_path + "/txt")
    checkpath(args.output_path + "/img")

    #transform = S3FDBasicTransform(cfg.INPUT_SIZE, cfg.MEANS)

    counter = 0

    img_paths = glob.glob("{}/*.png".format(args.input_path))
    print("There are {} images in input folder.".format(len(img_paths)))

    for img_path in img_paths:
        counter += 1
        print("Process {:04d} th image".format(counter))

        img_name = get_name(img_path)
        out_f = "{}/txt/{}.txt".format(args.output_path, img_name)
        out_img = "{}/img/{}.png".format(args.output_path, img_name)
        # if os.path.isfile(out_f):
        #     print("exists.")
        #     continue
        fout = open(out_f, "w")



        t1 = time.time()
        #img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = np.array(img)
        try:
            bboxes = detect_face(net, img, args.thresh)
            t2 = time.time()
            print('Detect costs %.4f' % (t2 - t1))
            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox
                fout.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                    "face", score, x1, y1, x2, y2))
            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(out_img, img)
        except Exception as e:
            print(e)
            print("Error.")

        fout.close()