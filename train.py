#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from data.config import cfg
from layers.modules import MultiBoxLoss
from data.darkface import DARKDetection, detection_collate
from models.factory import build_net, basenet_factory

import tensorboard_logger as tl
from tqdm import tqdm
import shutil
from models.DSFD_Unet import DSFD_Unet


parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Pytorch')
# train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--exp_name',
                    type = str,
                    help='The experiment name.')
parser.add_argument('--batch_size',
                    default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--model',
                    default='vgg', type=str,
                    choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--resume',
                    default= None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--dsfd_weights',
                    default= None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--unet_weights',
                    default= None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--fix_unet',
                    default=True, type=bool,
                    help='fix unet weights')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=False, type=bool,
                    help='Use mutil Gpu training')
# parser.add_argument('--save_folder',
#                     default='weights/',
#                     help='Directory for saving checkpoint models')
args = parser.parse_args()

if not args.multigpu:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


# make new dir
weights_path = "train_experiments/{}/weights".format(args.exp_name)
log_path = "train_experiments/{}/log".format(args.exp_name)
checkpath(weights_path)
checkpath(log_path)

#cfg.FACE.TRAIN_FILE = train/Lime_img/train/

train_dataset = DARKDetection(cfg.FACE.TRAIN_FILE, mode='train')
val_dataset = DARKDetection(cfg.FACE.VAL_FILE, mode='val')

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)
val_batchsize = args.batch_size//2 # args.batch_size - 1
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)

min_loss = np.inf
tl.configure(logdir=log_path, flush_secs=3)

def train():
    start_epoch = 0
    iteration = 0
    step_index = 0

    # basenet = basenet_factory(args.model)
    # dsfd_net = build_net('train', cfg.NUM_CLASSES, args.model)
    # net = dsfd_net

    net = DSFD_Unet(args.dsfd_weights, args.unet_weights, mode="train", fix_unet = args.fix_unet)


    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        # start_epoch = net.load_weights(args.resume)
        # iteration = start_epoch * per_epoch_size
        # net.load_state_dict(torch.load(args.resume))
        net = load_my_state_dict(net, torch.load(args.resume))
    #
    # else:
    #     # base_weights = torch.load(args.save_folder + basenet)
    #     # print('Load base network {}'.format(args.save_folder + basenet))
    #     if args.model == 'vgg':
    #         pass
    #         # net.vgg.load_state_dict(base_weights)
    #     else:
    #         net.resnet.load_state_dict(base_weights)

    dsfd_net = net

    if args.cuda:
        if args.multigpu:
            net = torch.nn.DataParallel(dsfd_net)
        net = net.cuda()
        cudnn.benckmark = True

    # if not args.resume:
    #     print('Initializing weights...')
    #     dsfd_net.extras.apply(dsfd_net.weights_init)
    #     dsfd_net.fpn_topdown.apply(dsfd_net.weights_init)
    #     dsfd_net.fpn_latlayer.apply(dsfd_net.weights_init)
    #     dsfd_net.fpn_fem.apply(dsfd_net.weights_init)
    #     dsfd_net.loc_pal1.apply(dsfd_net.weights_init)
    #     dsfd_net.conf_pal1.apply(dsfd_net.weights_init)
    #     dsfd_net.loc_pal2.apply(dsfd_net.weights_init)
    #     dsfd_net.conf_pal2.apply(dsfd_net.weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg, args.cuda)
    print('Loading wider dataset...')
    print('Using the specified args:')
    print(args)

    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            # t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2
            loss.backward()
            optimizer.step()
            # t1 = time.time()
            losses += loss.item()

            # if iteration % 10 == 0:
            #     tloss = losses / (batch_idx + 1)
                #print('Timer: %.4f' % (t1 - t0))
                #print('epoch:' + repr(epoch) + ' || iter:' +
                #      repr(iteration) + ' || Loss:%.4f' % (tloss))
                #print('->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
                #    loss_c_pal1.item(), loss_l_pa1l.item()))
                #print('->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
                #    loss_c_pal2.item(), loss_l_pa12.item()))
                #print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))

            # if iteration != 0 and iteration % 200 == 0:
            #     # print('Saving state, iter:', iteration)
            #     file = 'dsfd_' + repr(iteration) + '.pth'
            #     torch.save(dsfd_net.state_dict(),
            #                os.path.join(save_folder, file))
            iteration += 1

        val_loss = val(epoch, net, dsfd_net, criterion)

        print("Epoch {}, train_loss {:.3f}, val_loss {:.3f}".format(epoch, losses / len(train_loader), val_loss))
        tl.log_value("train_loss", losses / len(train_loader), step=epoch)
        tl.log_value("val_loss", val_loss, step=epoch)

        if iteration == cfg.MAX_STEPS:
            break


def val(epoch, net, dsfd_net, criterion):
    net.eval()
    step = 0
    losses = 0
    t1 = time.time()
    for batch_idx, (images, targets) in enumerate(val_loader):
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        out = net(images)
        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
        loss = loss_l_pa12 + loss_c_pal2
        losses += loss.item()
        step += 1

    tloss = losses / step
    t2 = time.time()
    print('Timer: %.4f' % (t2 - t1))
    print()
    print("-"*30)
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))
    print("-"*30)
    print()

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        torch.save(dsfd_net.state_dict(), os.path.join(
            weights_path, 'dsfd_best_[epoch]-{}-[val_loss]-{:.3f}.pth'.format(epoch, tloss)))
        min_loss = tloss

    torch.save(dsfd_net.state_dict(), os.path.join(weights_path, 'dsfd_checkpoint_last.pth'))

    return tloss


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_my_state_dict(model, pretrained_dict):
    model_dict = model.state_dict()
    print("model dict parms number", len(model_dict))
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    print("DSFD weights length: ", len(pretrained_dict))
    model_dict.update(pretrained_dict)
    print("model dict parms number", len(model_dict))
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return model


if __name__ == '__main__':
    train()
