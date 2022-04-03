#!/bin/bash

python metrlib/main.py \
  --dataset cars196 \
  --kernels 6 \
  --source . \
  --n_epochs 150 \
  --log_online \
  --project metric-learning \
  --group CARreg_Margin_b06_Distance \
  --loss_margin_beta 0.6 \
  --seed 0 \
  --gpu 0 \
  --bs 112 \
  --samples_per_class 2 \
  --loss margin \
  --batch_mining rho_distance \
  --arch resnet50_frozen_normalize \
  --miner_rho_distance_cp 0.35 \
  --save_path runs
