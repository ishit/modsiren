#!/bin/bash
TRAIN=demosaicnet/train/filelist.txt
VAL=demosaicnet/val/filelist.txt
python train_img.py --model modsiren --gpu 2 --train_filenames $TRAIN --val_filenames $VAL
