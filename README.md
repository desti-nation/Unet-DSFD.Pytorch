## weights
Link：https://share.weiyun.com/5Oib0Rw Password：pqx9mn

## train

on single gpu and train on unet weight and dsfd weight

```
CUDA_VISIBLE_DEVICES=1 python3 train.py --exp_name 02-weight=01last-lr=1e-3-data=orig --lr 1e-3 --dsfd_weights ./pretrained_weights/dsfd_vgg_0.880.pth --unet_weights ./pretrained_weights/unet_enhance.pth --batch_size 3 --resume ./train_experiments/01-weight=dsfd+unet-lr=1e-4-data=orig-keyboardinter35/weights/dsfd_checkpoint_last.pth
```

on multiple gpus and train on full weights
```
python3 train.py --exp_name 05-weight=04best-lr=1e-6-data=orig --lr 1e-6  --batch_size 20 --resume /home/lb/lb/DSFD.Unet.pytorch/train_experiments/04-weight=03last-lr=1e-4-data=orig/weights/dsfd_best_[epoch]-2-[val_loss]-4.538.pth --multigpu True
```

## test

```
CUDA_VISIBLE_DEVICES=0 python3 demo.py --save_dir /home/lb/lb/DSFD.Unet.pytorch/test_experiments/05-weight=04best-lr=1e-6-data=orig-thre=0.3  --model  /home/lb/lb/DSFD.Unet.pytorch/train_experiments/05-weight=04best-lr=1e-6-data=orig/weights/dsfd_best_[epoch]-6-[val_loss]-4.536.pth --thresh 0.3 --img_path /home/lb/lx/data/track2.2_test_sample --gt_dir /home/lb/lx/data/track2.2_test_label/
```

## calculate AP
>  AP code is provided at https://github.com/Ir1d/DARKFACE_eval_tools. We also provide an independent docker image at https://hub.docker.com/r/scaffrey/eval_tools. 



