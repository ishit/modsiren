#!/bin/bash
TRAIN=demosaicnet/train/filelist.txt
VAL=demosaicnet/val/filelist.txt
python train_img.py --model hyper --gpu 3 --train_filenames $TRAIN --val_filenames $VAL
