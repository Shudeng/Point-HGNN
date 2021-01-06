# model configs of hgnn_votehead

model = dict(
    type='HGNN_VoteHead',
    # backbone=dict(
    #     type='PointNet2SASSG',
    #     in_channels=4,
    #     num_points=(2048, 1024, 512, 256),
    #     radius=(0.2, 0.4, 0.8, 1.2),
    #     num_samples=(64, 32, 16, 16),
    #     sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
    #                  (128, 128, 256)),
    #     fp_channels=((256, 256), (256, 256)),
    #     norm_cfg=dict(type='BN2d'),
    #     sa_cfg=dict(
    #         type='PointSAModule',
    #         pool_mod='max',
    #         use_xyz=True,
    #         normalize_xyz=True)),
    bbox_coder=dict(
        type='PartialBinBasedBBoxCoder',
        num_sizes=3,  # 18
        num_dir_bins=1,
        with_rot=True,  # False
        # mean_sizes needs changes as dataset varies
        mean_sizes=[[0.76966727, 0.8116021, 0.92573744],
                    [1.876858, 1.8425595, 1.1931566],
                    [0.61328, 0.6148609, 0.7182701]]),
                    # [1.3955007, 1.5121545, 0.83443564],
                    # [0.97949594, 1.0675149, 0.6329687],
                    # [0.531663, 0.5955577, 1.7500148],
                    # [0.9624706, 0.72462326, 1.1481868],
                    # [0.83221924, 1.0490936, 1.6875663],
                    # [0.21132214, 0.4206159, 0.5372846],
                    # [1.4440073, 1.8970833, 0.26985747],
                    # [1.0294262, 1.4040797, 0.87554324],
                    # [1.3766412, 0.65521795, 1.6813129],
                    # [0.6650819, 0.71111923, 1.298853],
                    # [0.41999173, 0.37906948, 1.7513971],
                    # [0.59359556, 0.5912492, 0.73919016],
                    # [0.50867593, 0.50656086, 0.30136237],
                    # [1.1511526, 1.0546296, 0.49706793],
                    # [0.47535285, 0.49249494, 0.5802117]]),

    bbox_head=dict(
        # type='VoteHead',
        vote_module_cfg=dict(
            in_channels=64,  # 256
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(64, 256),  # (256, 256)
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=256,
            radius=0.3,
            num_sample=16,
            #mlp_channels=[256, 128, 128, 128],
            mlp_channels=[64, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True),
        pred_layer_cfg=dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=5.0),
        center_loss=dict(
            type='ChamferDistance',
            mode='l2',
            reduction='sum',
            loss_src_weight=10.0,
            loss_dst_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0 / 3.0),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote')
test_cfg = dict(
    sample_mod='seed', nms_thr=0.25, score_thr=0.05, per_class_proposal=True)

# The schedule is usually used by models trained on KITTI dataset

# The learning rate set in the cyclic schedule is the initial learning rate
# rather than the max learning rate. Since the target_ratio is (10, 1e-4),
# the learning rate will change from 0.0018 to 0.018, than go to 0.0018*1e-4
lr = 0.0018
# The optimizer follows the setting in SECOND.Pytorch, but here we use
# the offcial AdamW optimizer implemented by PyTorch.
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# We use cyclic learning rate and momentum schedule following SECOND.Pytorch
# https://github.com/traveller59/second.pytorch/blob/3aba19c9688274f75ebb5e576f65cfe54773c021/torchplus/train/learning_schedules_fastai.py#L69  # noqa
# We implement them in mmcv, for more details, please refer to
# https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327  # noqa
# https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130  # noqa
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
# Although the total_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual total epoch
# number could be Nx40
total_epochs = 40

# # optimizer
# # This schedule is mainly used by models on indoor dataset,
# # e.g., VoteNet on SUNRGBD and ScanNet
# lr = 0.008  # max learning rate
# optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
# optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# lr_config = dict(policy='step', warmup=None, step=[24, 32])
# # runtime settings
# total_epochs = 36


checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=30,  # 50
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
