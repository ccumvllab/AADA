# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add MIC options
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
uda = dict(
    type='AADA',
    source_only=False,
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    mask_mode=None,
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=0,
    mask_generator=None,
    debug_img_interval=100,
    print_grad_magnitude=False,
    g_scale = 0.5,
    c_scale = 0.8,
    c_reg_coef = 10,
    aug_lr = 1e-4,
    aug_weight_decay = 1e-2,
    update_augmodel_iter = 1,
    enable_c_aug = True,
    enable_g_aug = True,
    enable_wandb = True,
    cross_attention_block = 4,
    enable_cross_attention = True,
    name = 'cs2acdc_MIC_GT_Normal_AugUp1',
    lr = 6e-05,
    max_iters = 40000,
    epsilon = 0.,
    cross_dis_coef = 10,
    enable_augment_context = True,
    f_dim = 19,
    augmentation_init = "random",
    enable_cross_seg = False,
    enable_CjAug = False,
    adv_coef = 1,
)
use_ddp_wrapper = True
# CUDA_VISIBLE_DEVICES=1 python run_experiments.py --config configs/mic/gtaHR2csHR_aada_mic_hrda.py
