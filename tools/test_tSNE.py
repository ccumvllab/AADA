# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Modification of config and checkpoint to support legacy models
# - Add inference mode and HRDA output flag

import argparse
import os
import numpy as np
import random

import mmcv
import torch
from torchvision.utils import save_image
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from .t_SNE import RunTsne

trainId2name = {255: 'trailer',
                    0: 'road',
                    1: 'sidewalk',
                    2: 'building',
                    3: 'wall',
                    4: 'fence',
                    5: 'pole',
                    6: 'traffic light',
                    7: 'traffic sign',
                    8: 'vegetation',
                    9: 'terrain',
                    10: 'sky',
                    11: 'person',
                    12: 'rider',
                    13: 'car',
                    14: 'truck',
                    15: 'bus',
                    16: 'train',
                    17: 'motorcycle',
                    18: 'bicycle',
                    -1: 'license plate'}
trainId2color = {255: (0, 0, 110),
                    0: (128, 64, 128),
                    1: (244, 35, 232),
                    2: (70, 70, 70),
                    3: (102, 102, 156),
                    4: (190, 153, 153),
                    5: (153, 153, 153),
                    6: (250, 170, 30),
                    7: (220, 220, 0),
                    8: (107, 142, 35),
                    9: (152, 251, 152),
                    10: (70, 130, 180),
                    11: (220, 20, 60),
                    12: (255, 0, 0),
                    13: (0, 0, 142),
                    14: (0, 0, 70),
                    15: (0, 60, 100),
                    16: (0, 80, 100),
                    17: (0, 0, 230),
                    18: (119, 11, 32),
                    -1: (0, 0, 143)}


def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    cfg.data.val.pipeline[1]['img_scale'] = tuple(
        cfg.data.val.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--sample',
        type=int,
        default=1000)
    parser.add_argument(
        '--SelClass',
        type=str,
        default='all')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--inference-mode',
        choices=[
            'same',
            'whole',
            'slide',
        ],
        default='same',
        help='Inference mode.')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='Run inference on the test set')
    parser.add_argument(
        '--hrda-out',
        choices=['', 'LR', 'HR', 'ATT'],
        default='',
        help='Extract LR and HR predictions from HRDA architecture.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir',
        type=str, help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    domId2name = {
        0:'gta',
        1:'syn',
        2:'cs',
        3:'dark',
        4:'acdc'}

    if args.SelClass == 'all':
        selected_cls = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
                        'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    elif args.SelClass == 'easy':
        selected_cls = ['road', 'building', 'vegetation', 'sky', 'car', 'bus']
    elif args.SelClass == 'hard':
        selected_cls = ['wall', 'fence', 'terrain', 'rider', 'motorcycle', 'bicycle']
    elif args.SelClass == 'mix':
        selected_cls = ['fence', 'terrain', 'rider', 'road', 'sky', 'car']
        
    output_dir = f'./t-SNE/{args.show_dir}'
    adding_name = args.show_dir
    tsnecuda = False
    extention = '.png'
    duplication = 1
    plot_memory = False
    clscolor = True
    # domains2draw = ['gta', 'synthia', 'cityscapes', 'dark', 'acdc']
    domains2draw = ['gta', 'cs']
    # 指定需要进行t-SNE的域，即数据集

    tsne_runner = RunTsne(selected_cls=selected_cls,   # 选择可视化几个类别
                          domId2name=domId2name,       # 不同域的ID
                          trainId2name=trainId2name,   # 标签中每个ID所对应的类别
                          trainId2color=trainId2color, # 标签中每个ID所对应的颜色
                          output_dir=output_dir,       # 保存的路径
                          tsnecuda=tsnecuda,           # 是否使用tsnecuda，如果不使用tsnecuda就使用MulticoreTSNE
                          extention=extention,         # 保存图片的格式
                          duplication=duplication)     # 程序循环运行几次，即保存多少张结果图片
    
    # s_path = os.path.join(output_dir,f'img/source/')
    # if not os.path.isdir(s_path):
    #     os.makedirs(s_path)
    # t_path = os.path.join(output_dir,f'img/target/')
    # if not os.path.isdir(t_path):
    #     os.makedirs(t_path)


    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    # model = MMDataParallel(model, device_ids=[0])
    model.to('cuda')
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(args.sample)
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            s_in, s_gt = data['img'].data[0].to('cuda'), data['gt_semantic_seg'].data[0].to('cuda')
            t_in, t_gt = data['target_img'].data[0].to('cuda'), data['target_gt'].data[0].to('cuda')
            feature_s = model.extract_feat(s_in)[3]
            tsne_runner.input2basket(feature_s, s_gt, domains2draw[0])
            feature_t = model.extract_feat(t_in)[3]
            tsne_runner.input2basket(feature_t, t_gt, domains2draw[1])

            batch_size = len(feature_s)
            for _ in range(batch_size):
                prog_bar.update()
            if prog_bar.completed==prog_bar.task_num:
                break
            # means, stds = get_mean_std(data['img_metas'].data[0], 'cuda')
            # s_in = torch.clamp(denorm(s_in, means, stds), 0, 1)
            # t_in = torch.clamp(denorm(t_in, means, stds), 0, 1)
            # save_image(s_in,os.path.join(s_path,f'{i}.jpg'))
            # save_image(t_in,os.path.join(t_path,f'{i}.jpg'))
    del model
    tsne_runner.draw_tsne(domains2draw, adding_name=adding_name, plot_memory=plot_memory, clscolor=clscolor)
    return True


if __name__ == '__main__':
    main()
