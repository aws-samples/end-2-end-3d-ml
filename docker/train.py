# from mmcv import Config, DictAction
# from mmdet3d.apis import train_model
# from mmdet3d.utils import collect_env, get_root_logger
# from mmdet3d.apis import init_model
# from mmdet3d.datasets import build_dataset
# from mmdet3d.models import build_model

# Copyright (c) OpenMMLab. All rights reserved.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from __future__ import division

import argparse
import boto3
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from tools.data_converter import kitti_converter, kitti_data_utils

# Check Pytorch installation
import torch, torchvision
print('torch version:', torch.__version__, torch.cuda.is_available())
print('torchvision version:', torchvision.__version__)

# Check mmdet3d installation
import mmdet3d
print('mmdet3d version:', mmdet3d.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print('cuda version:', get_compiling_cuda_version())
print('compiler information:', get_compiler_version())

os.system('echo ----------')
os.system('nvidia-smi')
os.system('echo ----------')

os.environ['MASTER_PORT'] = '12345'
os.environ['MASTER_ADDR'] = 'algo-1'
try:
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
except:
    if "WORLD_SIZE" not in os.environ:
        os.environ['WORLD_SIZE'] = "1"
    if "RANK" not in os.environ:
        os.environ['RANK'] = "0"

#export WORLD_SIZE=OMPI_COMM_WORLD_SIZE
#export RANK=OMPI_COMM_WORLD_RANK
#export MASTER_ADDR=algo1
#export MASTER_PORT=12345

os.system('echo train folder contents')
os.system('ls /opt/ml/input/data/train')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--batch-size', help='training batch size', default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--distributed', type=int, default=0)
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--load-path', help='path to load model from')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    print('Rank:',os.environ['RANK'])
    # check /opt/ml/code contents
    os.system('ls /opt/ml/code')

    cfg = Config.fromfile(args.config)
    num_gpus = torch.cuda.device_count() # int(os.environ['SM_NUM_GPUS'])
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.load_path is not None:
        cfg.load_from = args.load_path

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(num_gpus) #range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        _, world_size = get_dist_info()

        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(world_size)
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        cfg.gpu_ids = range(world_size)

        
        print('world size:',os.environ['WORLD_SIZE'])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)
    cfg['data_root'] = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes'
    cfg.data.train.dataset.pipeline[4].db_sampler.data_root = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes'
    cfg.data.train.dataset.pipeline[4].db_sampler.info_path = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes/a2d2_dbinfos_train.pkl'
    cfg.data.train.dataset.data_root = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes'
    cfg.data.train.dataset.ann_file = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes/a2d2_infos_train.pkl'
    cfg.data.test.data_root = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes'
    cfg.data.test.ann_file = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes/a2d2_infos_test.pkl'
    cfg.data.val.data_root = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes'
    cfg.data.val.ann_file = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes/a2d2_infos_test.pkl'
    cfg.db_sampler.data_root = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes'
    cfg.db_sampler.info_path = '/opt/ml/input/data/train/camera_lidar_semantic_bboxes/a2d2_dbinfos_train.pkl'
    cfg.dataset_type = 'A2D2Dataset'
# need to set this to get it to register
    cfg.data.train.dataset.type = 'A2D2Dataset'
    cfg.data.test.type = 'A2D2Dataset'
    cfg.data.val.type = 'A2D2Dataset'
    # need to add dbinfos
    cfg.runner.max_epochs = int(args.epochs)
    cfg.data.samples_per_gpu = int(args.batch_size)
    
    print('trying to build dataset')
    datasets = [build_dataset(cfg.data.train)]

    print('test dataloader', datasets[0][0])
    
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    print('model', model)

#     os.system('ls /opt/ml/input/data/train')
    print('config:',cfg)
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()