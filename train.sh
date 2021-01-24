<<<<<<< Updated upstream
GPU_ID=2,3
=======
GPU_ID=0,1
>>>>>>> Stashed changes

## single gpu
#CUDA_VISIBLE_DEVICES=2 python train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py

## multi-gpu distribute train
<<<<<<< Updated upstream
CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node=2 --master_port=29555 train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py --launcher pytorch
=======
CUDA_VISIBLE_DIVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py --launcher pytorch
>>>>>>> Stashed changes

#python train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py

#python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py configs/_base_/datasets/kitti-3d-3class.py \
#    --head_type PlainHead

#python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py configs/_base_/datasets/kitti-3d-3class.py \
#    --head_type VoteHead --distributed

#python train.py configs/_base_/datasets/kitti-3d-3class.py \
#    --head_type VoteHead

#python train.py configs/_base_/datasets/kitti-3d-3class.py \
    #--head_type PlainHead

