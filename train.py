import torch
import os
import argparse
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader, replace_ImageToTensor
import copy
from hgnn_model import HGNN
from head.box_encoding import get_encoding_len

import time
import random
import numpy as np
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer)
# from mmcv.utils import build_from_arg
from mmdet.core import DistEvalHook, EvalHook
from mmdet.utils import get_root_logger

SINGLE = True

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--model_config', type=str, default='./configs/hgnn_votehead.py')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')  # 0
    parser.add_argument('--distributed', action='store_true', help='distributed')
    parser.add_argument('--gpu_ids', type=list, default=[0, 1], help="gpu_ids")
    parser.add_argument('--backend', type=str, default="nccl", help="backend")
    parser.add_argument('--launcher', type=str, default="pytorch", help="launcher")
    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--find_unused_parameters', type=bool, default=True)
    parser.add_argument('--find_unused_parameters', type=bool, default=False)

    parser.add_argument('--head_type', type=str, default='PlainHead', help='PlainHead or VoteHead')
    parser.add_argument('--box_encoding_method', type=str, default="classaware_all_class_box_encoding")
    parser.add_argument('--work-dir', type=str, default='./log', help='the dir to save logs and models')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    arg = Config.fromfile(args.config)
    # print('data_arg: ', arg)
    model_arg = Config.fromfile(args.model_config)
    # print('moder_arg: ', model_arg)
    arg.merge_from_dict(model_arg)
    # print(arg.model.bbox_head)
    # if args.options is not None:
    #     arg.merge_from_dict(args.options)

    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)
    arg.work_dir = args.work_dir
    logger = get_root_logger(arg.log_level)
    """
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        print("local_rank", args.local_rank)
    """

    train_dataset = build_dataset(arg.data.train)
    if not SINGLE:
        train_dataset = train_dataset if isinstance(train_dataset, (list, tuple)) else [train_dataset]

    # val_dataset = copy.deepcopy(arg.data.val)
    # val_dataset.pipeline = arg.data.train.pipeline
    # val_dataset = build_dataset(val_dataset)
    if args.distributed:
        dist_params = {"backend": args.backend}
        init_dist(args.launcher, **dist_params)

    arg.data.samples_per_gpu = 1
    print("samples_per_gpu", arg.data.samples_per_gpu)
    print("workers_per_gpu", arg.data.workers_per_gpu)

    if SINGLE:
        train_dataloader = build_dataloader(
            train_dataset,
            arg.data.samples_per_gpu,
            arg.data.workers_per_gpu,
            # arg.gpus will be ignored if distributed
            len(args.gpu_ids),
            dist=args.distributed,
            # dist=False,
            seed=args.seed)
    else:
        train_dataloader = [build_dataloader(
            ds,
            arg.data.samples_per_gpu,
            arg.data.workers_per_gpu,
            # arg.gpus will be ignored if distributed
            len(args.gpu_ids),
            dist=args.distributed,
            # dist=False,
            seed=args.seed) for ds in train_dataset]


    if SINGLE:
        # print(train_dataloader)
        itr = iter(train_dataloader)
        # print(next(itr))

    downsample_voxel_sizes = [[0.5, 0.5, 0.5], [0.8, 0.8, 0.8], [1.1, 1.1, 1.1]]
    inter_radius = [1.0, 1.5, 2.0]
    intra_radius = [1.5, 2.0, 2.5]

    max_num_neighbors = 256  # 32 is default, 256 is adopted in Point-GNN
    # (256 for training and all edges for inference)
    num_classes = len(arg.class_names)
    box_encoding_len = get_encoding_len(args.box_encoding_method)  # used in PlainHead
    head_type = args.head_type
    print("class names", arg.class_names)
    # print("box_encoding_len", box_encoding_len)
    model = HGNN(downsample_voxel_sizes, inter_radius, intra_radius,
                 max_num_neighbors, num_classes, head_type, box_encoding_len, **arg)  # **arg.model.bbox_head

    model = model.cuda()

    if args.distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=args.find_unused_parameters
        )
    else:
        model = MMDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
        )
    # model = model.cuda()

    if SINGLE:
        data = next(itr)
        print("data.keys", data.keys())
        points = data['points']
        img_metas = data['img_metas']
        gt_bboxes_3d = data['gt_bboxes_3d']
        gt_labels_3d = data['gt_labels_3d']  # .long()

        # print(len(points))
        model(points, img_metas, gt_bboxes_3d, gt_labels_3d)
        # print(losses)
    else:
        meta = None  # ?
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # build runner
        optimizer = build_optimizer(model, arg.optimizer)
        runner = EpochBasedRunner(
            model,
            optimizer=optimizer,
            work_dir=arg.work_dir,
            logger=logger,
            meta=meta)
        # an ugly workaround to make .log and .log.json filenames the same
        runner.timestamp = timestamp

        # fp16 setting
        fp16_arg = arg.get('fp16', None)
        if fp16_arg is not None:
            optimizer_config = Fp16OptimizerHook(
                **arg.optimizer_config, **fp16_arg, distributed=args.distributed)
        elif args.distributed and 'type' not in arg.optimizer_config:
            optimizer_config = OptimizerHook(**arg.optimizer_config)
        else:
            optimizer_config = arg.optimizer_config

        # register hooks
        runner.register_training_hooks(arg.lr_config, optimizer_config,
                                       arg.checkpoint_config, arg.log_config,
                                       arg.get('momentum_config', None))
        if args.distributed:
            runner.register_hook(DistSamplerSeedHook())

        validate = False
        # register eval hooks
        if validate:
            # Support batch_size > 1 in validation
            val_samples_per_gpu = arg.data.val.pop('samples_per_gpu', 1)
            if val_samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                arg.data.val.pipeline = replace_ImageToTensor(
                    arg.data.val.pipeline)
            val_dataset = build_dataset(arg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=val_samples_per_gpu,
                workers_per_gpu=arg.data.workers_per_gpu,
                dist=args.distributed,
                shuffle=False)
            eval_arg = arg.get('evaluation', {})
            eval_hook = DistEvalHook if args.distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_arg))

        # # user-defined hooks
        # if arg.get('custom_hooks', None):
        #     custom_hooks = arg.custom_hooks
        #     assert isinstance(custom_hooks, list), \
        #         f'custom_hooks expect list type, but got {type(custom_hooks)}'
        #     for hook_arg in arg.custom_hooks:
        #         assert isinstance(hook_arg, dict), \
        #             'Each item in custom_hooks expects dict type, but got ' \
        #             f'{type(hook_arg)}'
        #         hook_arg = hook_arg.copy()
        #         priority = hook_arg.pop('priority', 'NORMAL')
        #         hook = build_from_arg(hook_arg, HOOKS)
        #         runner.register_hook(hook, priority=priority)

        if arg.resume_from:
            runner.resume(arg.resume_from)
        elif arg.load_from:
            runner.load_checkpoint(arg.load_from)
        runner.run(train_dataloader, arg.workflow, arg.total_epochs)


if __name__ == "__main__":
    main()
