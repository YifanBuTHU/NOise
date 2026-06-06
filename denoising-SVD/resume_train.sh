#!/bin/bash
cd /home/BYF/Python/NOise/denoising-SVD
source /usr/local/anaconda3/bin/activate 3DUS
python train_n2n_complex.py \
    --resume experiments/complex_n2n/20260528-215406/checkpoints/model_epoch20.pth \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0003 \
    --comparison-freq 10 \
    --cuda 0 \
    >> experiments/complex_n2n/20260528-215406/resume_train.log 2>&1
