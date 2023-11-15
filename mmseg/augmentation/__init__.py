from . import augmentation_container
from . import cutout
from . import imagenet_augmentation
from . import nn_aug
# from . import replay_buffer

def build_augmentation(f_dim, g_use, c_use, t_use, g_scale, c_scale, c_reg_coef=0, with_condition=True, init='random', global_trans=False):
    c_aug, g_aug, t_aug = None, None, None
    if g_use:
        g_aug = nn_aug.GeometricAugmentation(f_dim, g_scale, with_condition=with_condition, init=init)
    if c_use:
        c_aug = nn_aug.ColorAugmentation(f_dim, c_scale, with_condition=with_condition, init=init, global_trans=global_trans)
    # 將自動擴增模型(g/c/t)包裝成方便更新的container
    augmentation = augmentation_container.AugmentationContainer(c_aug, g_aug, t_aug, c_reg_coef, f_dim)
    return augmentation
