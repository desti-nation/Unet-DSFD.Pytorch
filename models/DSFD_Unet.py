#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from layers import *
from data.config import cfg
from unet import *

from data.config import cfg
from layers.modules import MultiBoxLoss
from data.darkface import DARKDetection, detection_collate
from models.factory import build_net, basenet_factory
from unet import *


class DSFD_Unet(nn.Module):

    def __init__(self, dsfd_weights = None, unet_weights = None, mode = "train", fix_unet = True):
        super(DSFD_Unet, self).__init__()
        self.dsfd = build_net(mode, cfg.NUM_CLASSES)
        self.unet = UNet(3, 3)
        if dsfd_weights:
            self.dsfd.load_state_dict(torch.load(dsfd_weights))
        if unet_weights:
            self.unet.load_state_dict(torch.load(unet_weights))
        if fix_unet:
            for p in self.unet.parameters():
                p.requires_grad = False
        print("Fit params of unet")

    def forward(self, x):
        x = self.unet(x)
        x = self.dsfd(x)
        return x






