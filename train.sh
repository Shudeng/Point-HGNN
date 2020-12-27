#python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py configs/_base_/datasets/kitti-3d-3class.py \
#    --head_type PlainHead


python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py configs/_base_/datasets/kitti-3d-3class.py \
--head_type VoteHead

#python train.py configs/_base_/datasets/kitti-3d-3class.py \
    #--head_type VoteHead

#python train.py configs/_base_/datasets/kitti-3d-3class.py \
    #--head_type PlainHead

