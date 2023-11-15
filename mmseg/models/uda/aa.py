# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd
import torch.optim as optim
import torch.nn as nn

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio
from mmseg.core.ddp_wrapper import DistributedDataParallelWrapper
from ..losses import NonSaturatingLoss
from mmseg import augmentation
import wandb
from mmseg.ops import resize


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm

def linear_schedular(total, current, start_val, end_val):
    return start_val + (end_val-start_val)*current/total

def pow_schedular(total, current, start_val, end_val, pow):
    return start_val + (end_val-start_val)*current/total


@UDA.register_module()
class AADA(UDADecorator):

    def __init__(self, **cfg):
        super(AADA, self).__init__(**cfg)
        self.local_iter = 0
        self.debug_img_interval = cfg['debug_img_interval']
        self.max_iters = cfg['max_iters']
        self.source_only = cfg['source_only']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.update_augmodel_iter = cfg['update_augmodel_iter']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.enable_g_aug = cfg['enable_g_aug']
        self.enable_c_aug = cfg['enable_c_aug']
        self.enable_wandb = cfg['enable_wandb']
        self.cross_attention_block = cfg['cross_attention_block']
        self.cross_dis_coef = cfg['cross_dis_coef']
        self.enable_cross_attention = cfg['enable_cross_attention']
        self.enable_augment_context = cfg['enable_augment_context']
        self.seg_opit = None
        self.f_dim = cfg['f_dim']
        self.enable_cross_seg = cfg['enable_cross_seg']
        self.enable_CjAug = cfg['enable_CjAug']
        self.CjAug_prob = 1.
        self.adv_coef = cfg['adv_coef']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        self.adv_criterion = NonSaturatingLoss(cfg['epsilon'])
        self.trainable_aug = augmentation.build_augmentation(self.f_dim, cfg['enable_g_aug'], cfg['enable_c_aug'], cfg['g_scale'], cfg['c_scale'],
                                                        cfg['c_reg_coef'], with_condition=self.enable_augment_context)
        self.trainable_aug = self.trainable_aug.to('cuda:0')
        self.trainable_aug.train()
        self.aug_criterion  = nn.MSELoss()
        self.aug_optim = optim.AdamW(self.trainable_aug.parameters(), lr=cfg['aug_lr'], weight_decay=cfg['aug_weight_decay'])

        if self.enable_wandb:
            self.wandb_run = wandb.init(
                        # Set the project where this run will be logged
                        project="aada_mic_gta2cs",
                        name=cfg['name'],
                        # Track hyperparameters and run metadata
                        config={
                            "learning_rate": cfg['lr'],
                            "iters": cfg['max_iters'],
                            "pseudo_threshold": cfg['pseudo_threshold'],
                            "update_augmodel_iter": cfg['update_augmodel_iter'],
                            "aug_lr": cfg['aug_lr'],
                            "cross_attention_block": self.cross_attention_block,
                            "enable_cross_attention": self.enable_cross_attention,
                            'enable_augment_context': self.enable_augment_context,
                            "f_dim": self.f_dim,
                            "augmentation_init": cfg['augmentation_init'],
                            "enable_masking": self.enable_masking,
                            "enable_cross_seg": self.enable_cross_seg,
                            "enable_CjAug": self.enable_CjAug,
                            "adv_coef": self.adv_coef,
                        })

    def Normto01(self,x):
        x -= x.min(1,keepdim=True)[0]
        x /= x.max(1,keepdim=True)[0]
        return x

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        self.seg_opit = optimizer
        # optimizer.zero_grad()
        log_vars = self(**data_batch)
        # optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        
        if self.enable_wandb:
            self.wandb_run.log({
                                'iter': self.local_iter,
                                'seg lr': optimizer.param_groups[0]['lr'],})

        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def augment(self, img, gt, update=False, pw=None, only_caug=False):
        if self.enable_augment_context:
            if self.f_dim==19:
                feature = gt
            elif self.f_dim==512:
                with torch.no_grad():
                    feature = self.get_model().extract_feat(img)
                    feature = feature[3]
        else:
            feature = None
        if pw is not None:
            pw = pw.unsqueeze(dim=1)
        if update:
            aug_x, aug_x_t, swd, g_prob, c_prob, pw = self.trainable_aug(img, gt, feature, pw, update)
        else:
            with torch.no_grad():
                aug_x, aug_x_t, swd, g_prob, c_prob, pw = self.trainable_aug(img, gt, feature, pw, update, only_caug)
        if pw is not None:
            pw = pw.squeeze(dim=1)
        return aug_x, aug_x_t, swd, g_prob, c_prob, pw

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_gt,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        
        for p in self.trainable_aug.parameters():
            p.requires_grad = False
        for p in self.get_model().parameters():
            p.requires_grad = True

        self.update_debug_state()
        seg_debug = {}
    
        if self.enable_CjAug:
            self.CjAug_prob = linear_schedular(self.max_iters,self.local_iter, 1., 0.)
        use_CJ = True if random.uniform(0,1)<=self.CjAug_prob and self.enable_CjAug else False

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s if use_CJ else 0.0,
            'color_jitter_p': self.color_jitter_p  if use_CJ else 1.,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Augment source image
        if use_CJ:
            _, _, aug_img = strong_transform(strong_parameters, data=img.clone())
            s_g_prob, s_c_prob = None, None
            aug_gt_semantic_seg = gt_semantic_seg
        else:
            aug_img, aug_gt_semantic_seg, _, s_g_prob, s_c_prob, _ = self.augment(img, gt_semantic_seg)
        # Train on source images
        clean_losses = self.get_model().forward_train(
            aug_img, img_metas, aug_gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.enable_wandb:
            self.wandb_run.log({
                                'iter': self.local_iter,
                                'loss_source': clean_loss.item(),
                            })
            if self.enable_CjAug:
                self.wandb_run.log({
                                'iter': self.local_iter,
                                'CjAug_prob': self.CjAug_prob,
                            })
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        pseudo_label, pseudo_weight = None, None
        if not self.source_only:
            # Generate pseudo-label
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits = self.get_ema_model().generate_pseudo_label(
                target_img, target_img_metas)
            seg_debug['Target'] = self.get_ema_model().debug_output
            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
                ema_logits)
            del ema_logits
            pseudo_weight = self.filter_valid_pseudo_region(
                pseudo_weight, valid_pseudo_mask)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            # Apply mixing
            mixed_img, mixed_lbl, aug_mixed_img = [None] * batch_size, [None] * batch_size, [None] * batch_size
            mixed_seg_weight = pseudo_weight.clone()
            mix_masks = get_class_masks(gt_semantic_seg)
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': 0.0,
                'color_jitter_p': 1.,
                'blur': 0,
                'mean': means[0].unsqueeze(0),  # assume same normalization
                'std': stds[0].unsqueeze(0)
            }
            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i], _ = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img[i])),
                    target=torch.stack(
                        (gt_semantic_seg[i][0], pseudo_label[i])))
                _, mixed_seg_weight[i], _ = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            del gt_pixel_weight
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)

            # Augment target image
            if use_CJ:
                strong_parameters = {
                    'mix': None,
                    'color_jitter': random.uniform(0, 1),
                    'color_jitter_s': self.color_jitter_s if use_CJ else 0.0,
                    'color_jitter_p': self.color_jitter_p  if use_CJ else 1.,
                    'blur': random.uniform(0, 1) if self.blur else 0,
                    'mean': means[0].unsqueeze(0),  # assume same normalization
                    'std': stds[0].unsqueeze(0)
                }
                _, _, aug_mixed_img = strong_transform(strong_parameters, data=mixed_img.clone())
                aug_mixed_lbl, t_g_prob, t_c_prob = mixed_lbl, None, None
            else:
                aug_mixed_img, aug_mixed_lbl, _, t_g_prob, t_c_prob, mixed_seg_weight = self.augment(mixed_img, mixed_lbl,pw=mixed_seg_weight)
            # Train on mixed images
            mix_losses = self.get_model().forward_train(
                aug_mixed_img,
                img_metas,
                aug_mixed_lbl,
                seg_weight=mixed_seg_weight,
                return_feat=False,
            )
            seg_debug['Mix'] = self.get_model().debug_output
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()
        # Masked Training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            # aug_tar_img, aug_pseudo_label, _, _, _, _ = self.augment(target_img, pseudo_label, for_mask=True)
            masked_loss = self.mic(self.get_model(), img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, valid_pseudo_mask,
                                   pseudo_label, pseudo_weight)
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()
        
        if self.enable_cross_seg:
            c_aug_mixed_img, _, _, _, _, _ = self.augment(mixed_img, mixed_lbl,only_caug=True)
            s=0.5
            scale_mixed_img = resize(
                    input=mixed_img,
                    scale_factor=s,
                    mode='bilinear')
            scale_c_aug_mixed_img = resize(
                    input=c_aug_mixed_img,
                    scale_factor=s,
                    mode='bilinear')
            scale_img = resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear')
            cross_ori = self.get_model().forward_cross(scale_img, scale_mixed_img, 4)
            cross_aug = self.get_model().forward_cross(scale_img, scale_c_aug_mixed_img, 4)
            cross_dis = self.aug_criterion(cross_aug, cross_ori)
            loss_cross = self.cross_dis_coef*cross_dis
            loss_cross.backward()
            if self.local_iter % self.debug_img_interval == 0:
                crossA_out_dir = os.path.join(self.train_cfg['work_dir'],
                                    'seg_cross_attention_map')
                os.makedirs(crossA_out_dir, exist_ok=True)
                vis_source_img = torch.clamp(denorm(scale_img, means, stds), 0, 1)
                vis_trg_img = torch.clamp(denorm(scale_mixed_img, means, stds), 0, 1)
                vis_aug_trg_img = torch.clamp(denorm(scale_c_aug_mixed_img, means, stds), 0, 1)
                cross_ori_sum = self.Normto01(torch.sum(cross_ori,dim=1))
                cross_aug_sum = self.Normto01(torch.sum(cross_aug,dim=1))
                for j in range(batch_size):
                    rows, cols = 2, 4
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    # img
                    subplotimg(axs[0][0], cross_ori_sum[j], 'Ori map d_sum', cmap='gray')
                    subplotimg(axs[1][0], cross_aug_sum[j], 'Aug map d_sum', cmap='gray')
                    subplotimg(axs[0][1], cross_ori[j][0], 'Ori map d_1', cmap='gray')
                    subplotimg(axs[1][1], cross_aug[j][0], 'Aug map d_1', cmap='gray')
                    subplotimg(axs[0][2], vis_trg_img[j], 'Ori Target Image')
                    subplotimg(axs[1][2], vis_aug_trg_img[j], 'Aug Target Image')
                    subplotimg(axs[0][3], vis_source_img[j], 'Source Image')
                    for ax in axs.flat:
                            ax.axis('off')
                    plt.title = f"dis: {cross_dis.item()}"
                    plt.savefig(
                        os.path.join(crossA_out_dir,
                                    f'{(self.local_iter + 1):06d}_{j}.png'))
                    plt.close()
            del scale_img, scale_c_aug_mixed_img, scale_mixed_img

        self.seg_opit.step()
        self.seg_opit.zero_grad()

        if self.enable_wandb:
            self.wandb_run.log({
                                'iter': self.local_iter,
                                'loss_mix': mix_loss.item(),
                                'log_vars': log_vars,
                            })
            if self.enable_masking:
                self.wandb_run.log({
                                    'iter': self.local_iter,
                                    'loss_masked': masked_loss.item(),
                                })
            if self.enable_cross_seg:
                self.wandb_run.log({
                                    'iter': self.local_iter,
                                    'loss_cross_seg': loss_cross.item(),
                                })

        if self.local_iter % self.debug_img_interval == 0 and \
                not self.source_only:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'augment_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_source_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_aug_source_img = torch.clamp(denorm(aug_img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            vis_aug_trg_img = torch.clamp(denorm(aug_mixed_img, means, stds), 0, 1)
            with torch.no_grad():
                vis_pred_source = self.get_model().encode_decode(aug_img, gt_semantic_seg)
                vis_pred_source = vis_pred_source.argmax(dim=1)
                vis_pred_target = self.get_model().encode_decode(aug_mixed_img, target_img_metas)
                vis_pred_target = vis_pred_target.argmax(dim=1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                # img
                subplotimg(axs[0][0], vis_source_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                # aug img
                subplotimg(axs[0][1], vis_aug_source_img[j], f'Aug Source g:{s_g_prob[j] if s_g_prob is not None else -1}, c:{s_c_prob[j] if s_c_prob is not None else -1}')
                subplotimg(axs[1][1], vis_aug_trg_img[j], f'Aug Target g:{t_g_prob[j] if t_g_prob is not None else -1}, c:{t_c_prob[j] if t_c_prob is not None else -1}')
                # GT
                subplotimg(
                    axs[0][2],
                    aug_gt_semantic_seg[j],
                    'Aug_Source GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][2],
                    target_gt[j],
                    'Target GT',
                    cmap='cityscapes')
                # pseudo weight
                subplotimg(
                    axs[0][3], mixed_seg_weight[j], f'Pseudo W.,T={self.pseudo_threshold}', vmin=0, vmax=1)
                # target pseudo label
                subplotimg(
                    axs[1][3], aug_mixed_lbl[j], 'Target pseudo label', cmap='cityscapes')
                # pred mask
                subplotimg(
                    axs[0][4], vis_pred_source[j], 'Pred Source Seg', cmap='cityscapes')
                subplotimg(
                    axs[1][4], vis_pred_target[j], 'Pred Target Seg', cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'mic_hrda_debug')
            os.makedirs(out_dir, exist_ok=True)
            if seg_debug['Source'] is not None and seg_debug:
                if 'Target' in seg_debug:
                    seg_debug['Target']['Pseudo W.'] = mixed_seg_weight.cpu(
                    ).numpy()
                for j in range(batch_size):
                    cols = len(seg_debug)
                    rows = max(len(seg_debug[k]) for k in seg_debug.keys())
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(5 * cols, 5 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                        squeeze=False,
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            subplotimg(
                                axs[k2][k1],
                                **prepare_debug_out(f'{n1} {n2}', out[j],
                                                    means, stds))
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_{j}_s.png'))
                    plt.close()
                del seg_debug

        # Train Augmentation model
        if self.local_iter % self.update_augmodel_iter == 0:
            self.aug_optim.zero_grad()
            # stop seg model grad
            for p in self.trainable_aug.parameters():
                p.requires_grad = True
            for p in self.get_model().parameters():
                p.requires_grad = False
            # update aug model with source image
            # augmentation
            aug_img, aug_gt_semantic_seg, s_swd, _, _, _ = self.augment(img, gt_semantic_seg, update=True)
            # pred aug source image
            adv_pred = self.get_model().aug_stu_forward_train(aug_img, img_metas)
            loss_stu = self.adv_coef * self.adv_criterion(adv_pred, aug_gt_semantic_seg)
            clean_losses = self.get_ema_model().forward_train(aug_img, img_metas, aug_gt_semantic_seg)
            loss_tea, _ = self._parse_losses(clean_losses)
            loss_adv = loss_stu + loss_tea + s_swd
            loss_adv.backward()
            # cul aug_grad
            params = self.trainable_aug.parameters()
            s_aug_grads = [
                p.grad.detach() for p in params if p.grad is not None
            ]
            s_aug_grad = calc_grad_magnitude(s_aug_grads)

            if not self.enable_cross_attention:
                self.aug_optim.step()
            # update aug model with target image
            # augmentation
            aug_mixed_img, aug_mixed_lbl, t_swd, _, _, _ = self.augment(mixed_img, mixed_lbl, update=True)
            cross_ori = self.get_model().forward_cross(img, mixed_img, self.cross_attention_block)
            cross_aug = self.get_model().forward_cross(img, aug_mixed_img, self.cross_attention_block)
            cross_dis = self.cross_dis_coef * self.aug_criterion(cross_aug, cross_ori)
            loss_cross = cross_dis + t_swd
            if self.enable_cross_attention:
                loss_cross.backward()
                # cul aug_grad
                params = self.trainable_aug.parameters()
                s_aug_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                t_aug_grad = calc_grad_magnitude(s_aug_grads)
                self.aug_optim.step()
                # plot cross attention map
            if self.local_iter % self.debug_img_interval == 0:
                crossA_out_dir = os.path.join(self.train_cfg['work_dir'],
                                    'cross_attention_map')
                os.makedirs(crossA_out_dir, exist_ok=True)
                vis_source_img = torch.clamp(denorm(img, means, stds), 0, 1)
                vis_aug_source_img = torch.clamp(denorm(aug_img, means, stds), 0, 1)
                vis_trg_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
                vis_aug_trg_img = torch.clamp(denorm(aug_mixed_img, means, stds), 0, 1)
                cross_ori_sum = self.Normto01(torch.sum(cross_ori,dim=1))
                cross_aug_sum = self.Normto01(torch.sum(cross_aug,dim=1))
                for j in range(batch_size):
                    rows, cols = 2, 4
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    # img
                    subplotimg(axs[0][0], cross_ori_sum[j], 'Ori map d_sum', cmap='gray')
                    subplotimg(axs[1][0], cross_aug_sum[j], 'Aug map d_sum', cmap='gray')
                    subplotimg(axs[0][1], cross_ori[j][0], 'Ori map d_1', cmap='gray')
                    subplotimg(axs[1][1], cross_aug[j][0], 'Aug map d_1', cmap='gray')
                    subplotimg(axs[0][2], vis_trg_img[j], 'Ori Target Image')
                    subplotimg(axs[1][2], vis_aug_trg_img[j], 'Aug Target Image')
                    subplotimg(axs[0][3], vis_source_img[j], 'Source Image')
                    subplotimg(axs[1][3], vis_aug_source_img[j], 'Aug Source Image')
                    for ax in axs.flat:
                            ax.axis('off')
                    plt.title = f"dis: {cross_dis.item()}"
                    plt.savefig(
                        os.path.join(crossA_out_dir,
                                    f'{(self.local_iter + 1):06d}_{j}.png'))
                    plt.close()
            
            # update wandb log
            if self.enable_wandb:
                g_histograms = {}
                c_histograms = {}
                if self.enable_c_aug:
                    for tag, value in self.trainable_aug.c_aug.named_parameters():
                        tag = tag.replace('/', '.')
                        c_histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if self.enable_g_aug:
                    for tag, value in self.trainable_aug.g_aug.named_parameters():
                        tag = tag.replace('/', '.')
                        g_histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                self.wandb_run.log({
                                'iter': self.local_iter,
                                'aug lr': self.aug_optim.param_groups[0]['lr'],
                                'loss_stu': loss_stu.item(),
                                'loss_tea': loss_tea.item(),
                                's_color_reg': s_swd.item(),
                                'aug_adv_loss': loss_adv.item(),
                                's_aug_grad': s_aug_grad,
                                'cross_dis': cross_dis.item(),
                                't_color_reg': t_swd.item(),
                                'aug_corss_loss': loss_cross.item(),
                                'c_histograms': c_histograms,
                                'g_histograms': g_histograms,
                            })
                if self.enable_cross_attention:
                    self.wandb_run.log({
                                'iter': self.local_iter,
                                't_aug_grad': t_aug_grad,
                            })
        self.local_iter += 1
        return log_vars
