## single gpu
#python train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py

## multi-gpu distribute train
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py

#python train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py

#python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py configs/_base_/datasets/kitti-3d-3class.py \
#    --head_type PlainHead

#python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py configs/_base_/datasets/kitti-3d-3class.py \
#    --head_type VoteHead --distributed

#python train.py configs/_base_/datasets/kitti-3d-3class.py \
#    --head_type VoteHead

#python train.py configs/_base_/datasets/kitti-3d-3class.py \
    #--head_type PlainHead

