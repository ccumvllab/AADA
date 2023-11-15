import torch
import torch.nn as nn
from mmseg.ops import resize
import torchvision.transforms.functional as tf
import PIL

def slicd_Wasserstein_distance(x1, x2, n_projection=128):
    x1 = x1.flatten(-2).transpose(1, 2).contiguous() # (b, 3, h, w) -> (b, n, 3)
    x2 = x2.flatten(-2).transpose(1, 2).contiguous()
    rand_proj = torch.randn(3, n_projection, device=x1.device)
    rand_proj = rand_proj / (rand_proj.norm(2, dim=0, keepdim=True) + 1e-12)
    sorted_proj_x1 = torch.matmul(x1, rand_proj).sort(0)[0]
    sorted_proj_x2 = torch.matmul(x2, rand_proj).sort(0)[0]
    return (sorted_proj_x1 - sorted_proj_x2).pow(2).mean()

class AugmentationContainer(nn.Module):
    def __init__(
            self, c_aug=None, g_aug=None, t_aug=None, c_reg_coef=0, f_dim=512):
        super().__init__()
        self.c_aug = c_aug
        self.g_aug = g_aug
        self.t_aug = t_aug
        # lambda_swd
        self.c_reg_coef = c_reg_coef
        # 原圖在擴增時縮放的倍率，用於節省空間，使用pixel-wise擴增時必須要用，使用傳統擴增則還需要實驗看看
        self.scale_factor = 0.25
        self.f_dim = f_dim

    def get_params(self, x, c, c_aug, g_aug, t_aug, update):
        '''
        取得每個pixel的擴增參數
        '''
        # sample noise vector from unit gauss
        B,C,H,W = x.shape
        grid, scale, shift, g_prob, c_prob, p_stratgy, m_stratgy = [None] * B, [None] * B, [None] * B, [None] * B, [None] * B, [None] * B, [None] * B
        noise = x.new(x.shape[0], 128).normal_()
        target = x.clone()
        if c is not None:
            condition = c.clone()
        else:
            condition = c
        with torch.no_grad():
            target = resize(
                    input=target,
                    scale_factor=self.scale_factor,
                    mode='bilinear')
            if self.f_dim==19 and c is not None:
                condition = tf.resize(condition, [256,256], interpolation = PIL.Image.NEAREST)
        # sample augmentation parameters
        if g_aug is not None:
            grid, g_prob = g_aug(x, noise, c, update)
        if c_aug is not None:
            scale, shift, c_prob = c_aug(target, noise, condition, update)
            scale = resize(input=scale, scale_factor=1/self.scale_factor, mode='nearest')
            shift = resize(input=shift, scale_factor=1/self.scale_factor, mode='nearest')
        if t_aug is not None:
            p_stratgy, m_stratgy = t_aug(noise, condition)
        return (scale, shift), grid, g_prob, c_prob, p_stratgy, m_stratgy
    
    def get_params_for_mask(self, x, c, c_aug, update):
        '''
        取得對於mask data的pixel-wise擴增參數,目前不使用。
        '''
        # sample noise vector from unit gauss
        B,C,H,W = x.shape
        scale, shift, c_prob = None, None, None
        noise = x.new(x.shape[0], 128).normal_()
        target = x.clone()
        condition = c.clone()
        with torch.no_grad():
            target = resize(
                    input=target,
                    size=[256,256],
                    mode='bilinear')
            if self.f_dim==19:
                condition = tf.resize(condition, [256,256], interpolation = PIL.Image.NEAREST)
        # sample augmentation parameters
        scale, shift, c_prob = c_aug(target, noise, condition, update)
        scale = resize(input=scale, size=[H,W], mode='nearest')
        shift = resize(input=shift, size=[H,W], mode='nearest')
        return (scale, shift), c_prob

    def augmentation(self, x, x_t, condition, means, stds, c_aug, g_aug, t_aug, pw=None, update=False, for_mask=False):
        '''
        對影像進行擴增
        1. 先藉由擴增模型(g/c/t)取的擴增參數(pixel-wise or tradition)
        2. 進行擴增，在訓練擴增模型時會計算SWD，此時"update"會設為True
        '''
        g_prob, c_prob, p_stratgy, m_stratgy = None, None, None, None
        if for_mask:
            c_param, c_prob = self.get_params_for_mask(x, condition, c_aug, True)
        else:
            c_param, g_param, g_prob, c_prob, p_stratgy, m_stratgy = self.get_params(x, condition, c_aug, g_aug, t_aug, update)
        aug_x = x
        aug_x_t = x_t
        swd = torch.zeros(1, device=x.device)
        if t_aug is not None:
            aug_x, aug_x_t, pw = t_aug.transform(x, x_t, means, stds, p_stratgy, m_stratgy, pw)
            # color regularization
            if update and self.c_reg_coef > 0:
                swd = self.c_reg_coef * slicd_Wasserstein_distance(x, aug_x)
            else:
                swd = torch.zeros(1, device=x.device)
        else:
            if c_aug is not None:
                # color augmentation
                aug_x = c_aug.transform(x, *c_param)
                # color regularization
                if update and self.c_reg_coef > 0:
                    swd = self.c_reg_coef * slicd_Wasserstein_distance(x, aug_x)
                else:
                    swd = torch.zeros(1, device=x.device)
            if g_aug is not None and not for_mask:
                # geometric augmentation
                aug_x, aug_x_t, pw = g_aug.transform(aug_x, x_t, g_param, pw)
        return aug_x, aug_x_t, swd, g_prob, c_prob, pw, p_stratgy, m_stratgy

    def forward(self, x, x_t, condition, means, stds, pw=None, update=False, for_mask=False):
        x, x_t, swd, g_prob, c_prob, pw, p_stratgy, m_stratgy = self.augmentation(x, x_t, condition, means, stds, self.c_aug, self.g_aug, self.t_aug, pw, update=update, for_mask=for_mask)
        return x, x_t, swd, g_prob, c_prob, pw, p_stratgy, m_stratgy
