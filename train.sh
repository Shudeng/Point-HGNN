python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 train.py configs/_base_/datasets/kitti-3d-3class.py

