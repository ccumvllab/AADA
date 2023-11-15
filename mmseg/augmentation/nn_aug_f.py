import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def relaxed_bernoulli(logits, temp=0.0001, device='cpu'):
    u = torch.rand_like(logits, device=device)
    l = torch.log(u) - torch.log(1 - u)
    return ((l + logits)/temp).sigmoid().to(int)

class ColorAugmentation(nn.Module):
    def __init__(self, f_dim=512, scale=1, hidden=128, n_dim=128, dropout_ratio=0.8, with_condition=True, init='random'):
        super().__init__()
        
        conv = lambda ic, io, k : nn.Conv2d(ic, io, k, padding=k//2, bias=False)
        depthwise = lambda ic, io, k : nn.Conv2d(ic, io, kernel_size=k, padding=k//2, groups=ic, bias=False)
        pointwise = lambda ic, io : nn.Conv2d(ic, io, kernel_size=1, bias=False)
        bn2d = lambda c : nn.BatchNorm2d(c, track_running_stats=False)
        pool = lambda k, s : nn.MaxPool2d(kernel_size=k, stride=s)

        # embedding layer for RGB, condition, noise
        if with_condition:
            # 將輸入的條件進行降維 512*16*16->64*2*2
            self.context_enc_body = nn.Sequential(
                depthwise(f_dim, f_dim, 1),
                pointwise(f_dim,256),
                bn2d(256),
                nn.LeakyReLU(0.2, True),
                pool(4, 2),
                depthwise(256, 256, 1),
                pointwise(256,64),
                bn2d(64),
                nn.LeakyReLU(0.2, True),
                pool(4, 3),
            )
            # 將RGB, 條件, noise壓縮成256個channel
            self.color_enc1 = conv(3+256+n_dim, hidden, 1)
        else:
            self.color_enc1 = conv(3+n_dim, hidden, 1)
        
        # body for scale and shift
        self.color_enc_body = nn.Sequential(
            bn2d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            conv(hidden, hidden, 1),
            bn2d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Sequential()
        )
        # output layer for scale and shift
        self.c_regress = conv(hidden, 6, 1)

        self.register_parameter('logits', nn.Parameter(torch.zeros(1)))
        # initialize parameters
        self.reset(init)

        self.with_condition = with_condition
        self.scale = scale
        self.relax = True
        self.stochastic = True

    #目前擴增的機率是固定的，之後可以嘗試讓他可學習
    def sampling(self, scale, shift, temp=0.0001):
        prob = torch.ones([scale.shape[0]])
        if self.stochastic: # random apply
            logits = self.logits.repeat(scale.shape[0]).reshape(-1, 1, 1, 1)
            prob = relaxed_bernoulli(logits, temp, device=scale.device)
            if not self.relax: # hard sampling
                prob = (prob > 0.5).float()
            scale = 1 - prob + prob * scale
            shift = prob * shift # omit "+ (1 - prob) * 0"
        return scale, shift, prob.squeeze().to(int)

    def forward(self, x, noise, c=None, update=False):
        B,C,H,W = x.shape
        # per-pixel scale and shift "with" context information
        if self.with_condition:
            # 條件降維 512*16*16->64*2*2
            c = self.context_enc_body(c)
            # 攤平 64*2*2->256*1*1
            c = torch.flatten(c, 1)
            # 依照channel複製成 256*256*256
            c = c.view(B, c.shape[1], 1, 1).repeat(1,1,H,W)
            noise = noise.view(B,noise.shape[1],1,1).repeat(1,1,H,W)
            # 將每個pixel, 條件與noise concat
            feature = self.color_enc1(torch.cat([x, c, noise], 1))
        else: # per-pixel scale and shift "without" context information
            noise = noise.view(B,noise.shape[1],1,1).repeat(1,1,H,W)
            feature = self.color_enc1(torch.cat([x, noise], 1))
        feature = self.color_enc_body(feature)
        factor = self.c_regress(feature)
        # add up parameters
        scale, shift = factor.chunk(2, dim=1)
        scale = scale.sigmoid()
        shift = shift.sigmoid()
        # scaling
        scale = self.scale * (scale - 0.5) + 1
        shift = shift - 0.5
        # random apply
        if update:
            prob = torch.ones([scale.shape[0]])
        else:
            scale, shift, prob = self.sampling(scale, shift)

        return scale, shift, prob

    def reset(self, init='random'):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if init == 'random':
            nn.init.normal_(self.c_regress.weight)
        elif init == 'constant':
            nn.init.constant_(self.c_regress.weight, 0)
        nn.init.constant_(self.logits, 0)

    def transform(self, x, scale, shift):
        # ignore zero padding region
        with torch.no_grad():
            h, w = x.shape[-2:]
            mask = (x.sum(1, keepdim=True) == 0).float() # mask pixels having (0, 0, 0) color
            mask = torch.logical_and(mask.sum(-1, keepdim=True) < w,
                                     mask.sum(-2, keepdim=True) < h) # mask zero padding region

        x = (scale * x + shift) * mask
        return x
        

class GeometricAugmentation(nn.Module):
    def __init__(self, f_dim=512, scale=0.5, n_dim=128, dropout_ratio=0.8, with_condition=True, init='random'):
        super().__init__()

        hidden = n_dim
        linear = lambda ic, io : nn.Linear(ic, io, False)
        depthwise = lambda ic, io, k : nn.Conv2d(ic, io, kernel_size=k, padding=k//2, groups=ic, bias=False)
        pointwise = lambda ic, io : nn.Conv2d(ic, io, kernel_size=1, bias=False)
        bn1d = lambda c : nn.BatchNorm1d(c, track_running_stats=False)
        bn2d = lambda c : nn.BatchNorm2d(c, track_running_stats=False)
        pool = lambda k, s : nn.MaxPool2d(kernel_size=k, stride=s)

        if with_condition:
            self.context_enc_body = nn.Sequential(
                depthwise(f_dim, f_dim, 1),
                pointwise(f_dim,256),
                bn2d(256),
                nn.LeakyReLU(0.2, True),
                pool(4, 2),
                depthwise(256, 256, 1),
                pointwise(256,64),
                bn2d(64),
                nn.LeakyReLU(0.2, True),
                pool(4, 3),
            )

        self.body = nn.Sequential(
            linear(n_dim + 256 if with_condition else n_dim, hidden),
            bn1d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            linear(hidden, hidden),
            bn1d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
        )
        self.regressor = linear(hidden, 6)
        # identity matrix
        self.register_buffer('i_matrix', torch.Tensor([[1, 0, 0], [0, 1, 0]]).reshape(1, 2, 3))

        self.register_parameter('logits', nn.Parameter(torch.zeros(1)))
        # initialize parameters
        self.reset(init)

        self.with_condition = with_condition
        self.scale = scale

        self.relax = True
        self.stochastic = True

    def sampling(self, A, temp=0.0001):
        if self.stochastic: # random apply
            logits = self.logits.repeat(A.shape[0]).reshape(-1, 1, 1)
            prob = relaxed_bernoulli(logits, temp, device=logits.device)
            if not self.relax: # hard sampling
                prob = (prob > 0.5).float()
            return ((1 - prob) * self.i_matrix + prob * A), prob.squeeze().to(int)
        else:
            return A

    def forward(self, x, noise, c=None, update=False):
        if self.with_condition:
            c = self.context_enc_body(c)
            c = torch.flatten(c, 1)
            features = torch.cat((c, noise), dim=1)
        else:
            features = noise
        features = self.body(features)
        A = self.regressor(features).reshape(-1, 2, 3)
        # scaling
        A = self.scale * (A.sigmoid() - 0.5) + self.i_matrix
        # random apply
        if update:
            prob = torch.ones([A.shape[0]])
        else:
            A, prob = self.sampling(A)
        # matrix to grid representation
        grid = nn.functional.affine_grid(A, x.shape)
        return grid, prob

    def reset(self, init='random'):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # zero initialization
        if init=='random':
            nn.init.normal_(self.regressor.weight)
        elif init=='constant':
            nn.init.constant_(self.regressor.weight, 0)
        nn.init.constant_(self.logits, 0)

    def transform(self, x, x_t, grid, pw=None):
        x = F.grid_sample(x, grid, mode='bilinear')
        with torch.no_grad():
            ones_mask = torch.ones_like(x_t)
            inv_ones_mask = F.grid_sample(ones_mask.to(torch.float), grid, mode='nearest', padding_mode="zeros") - 1
            inv_color_mask = inv_ones_mask * (-255)
            x_t = F.grid_sample(x_t.to(torch.float), grid, mode='nearest', padding_mode="zeros") + inv_color_mask
            if pw is not None:
                pw = F.grid_sample(pw.to(torch.float), grid, mode='nearest', padding_mode="zeros")
        return x, x_t.to(torch.long), pw
