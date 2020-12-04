import torch
import os
import argparse
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from mmcv.parallel import MMDistributedDataParallel
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader
import copy
from hgnn_model import HGNN
from head.box_encoding import get_encoding_len


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--model_config', type=str, default='./configs/hgnn_votehead.py')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')  # 0
    parser.add_argument('--distributed', type=bool, default=True, help='distributed')
    parser.add_argument('--gpu_ids', type=list, default=[0, 1], help="gpu_ids")
    parser.add_argument('--backend', type=str, default="nccl", help="backend")
    parser.add_argument('--launcher', type=str, default="pytorch", help="launcher")
    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--find_unused_parameters', type=bool, default=True)
    parser.add_argument('--find_unused_parameters', type=bool, default=False)

    parser.add_argument('--head_type', type=str, default='PlainHead', help='PlainHead or VoteHead')
    parser.add_argument('--box_encoding_method', type=str, default="classaware_all_class_box_encoding")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # print('data_cfg: ', cfg)
    model_cfg = Config.fromfile(args.model_config)
    # print('moder_cfg: ', model_cfg)
    cfg.merge_from_dict(model_cfg)
    # print(cfg.model.bbox_head)
    # if args.options is not None:
    #     cfg.merge_from_dict(args.options)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        print("local_rank", args.local_rank)

    train_dataset = build_dataset(cfg.data.train)

    # val_dataset = copy.deepcopy(cfg.data.val)
    # val_dataset.pipeline = cfg.data.train.pipeline
    # val_dataset = build_dataset(val_dataset)
    if args.distributed:
        dist_params = {"backend": args.backend}
        init_dist(args.launcher, **dist_params)

    cfg.data.samples_per_gpu = 1
    print("samples_per_gpu", cfg.data.samples_per_gpu)
    print("workers_per_gpu", cfg.data.workers_per_gpu)

    train_dataloader = build_dataloader(
        train_dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # cfg.gpus will be ignored if distributed
        len(args.gpu_ids),
        dist=args.distributed,
        seed=args.seed)

    # print(train_dataloader)
    itr = iter(train_dataloader)
    # print(next(itr))

    downsample_voxel_sizes = [[0.5, 0.5, 0.5], [0.8, 0.8, 0.8], [1.1, 1.1, 1.1]]
    inter_radius = [1.0, 1.5, 2.0]
    intra_radius = [1.5, 2.0, 2.5]

    max_num_neighbors = 256  # 32 is default, 256 is adopted in Point-GNN
    # (256 for training and all edges for inference)
    num_classes = len(cfg.class_names)
    box_encoding_len = get_encoding_len(args.box_encoding_method)  # used in PlainHead
    head_type = args.head_type
    print("class names", cfg.class_names)
    # print("box_encoding_len", box_encoding_len)
    model = HGNN(downsample_voxel_sizes, inter_radius, intra_radius,
                 max_num_neighbors, num_classes, head_type, box_encoding_len, **cfg)  # **cfg.model.bbox_head

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=args.find_unused_parameters
    )

    data = next(itr)
    print("data.keys", data.keys())
    points = data['points']
    img_metas = data['img_metas']
    gt_bboxes_3d = data['gt_bboxes_3d']
    gt_labels_3d = data['gt_labels_3d']

    model(points, img_metas, gt_bboxes_3d, gt_labels_3d)


if __name__ == "__main__":
    main()
