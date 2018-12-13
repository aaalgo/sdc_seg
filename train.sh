#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
../aardvark/train-fcn-slim.py \
                     --classes 4 \
                     --db scratch/cityscape/train.db \
                     --val_db scratch/cityscape/val.db \
                     --val_epochs 10 \
                     --net resnet_v2_50 \
                     --batch 4 \
                     --max_size 1024 \
                     --clip_shift 16 \
                     --fix_width 1024 \
                     --fix_height 512 \
                     --ckpt_epochs 1 \
                     --nocache \
                     --patch_slim \
                     $*
