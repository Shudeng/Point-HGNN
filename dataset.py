import os
import torch
from mmdet3d.datasets import build_dataset
from mmdet3d.ops.voxel.voxel_layer import dynamic_voxelize

class Dataset():
    def __init__(self, dataset_cfg, radius_list, point_cloud_range):
        self.dataset = build_dataset(dataset_cfg)
        self.radius_list = radius_list
        self.point_cloud_range = point_cloud_range

    def voxelize(self, points, voxel_size):
        print(points)
        print("points.shape", points.shape)

        coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
        dynamic_voxelize(points[:3], coors, voxel_size, self.point_cloud_range, 3)
        print("res", coors)
        print("res shape", coors.shape)

        sums = coors.sum(1)
        print(sums)
        pass

    def construct_graph(self, points, voxelize):
        pass



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        res = self.dataset.__getitem__(idx)
        points = res['points']
        gt_bboxes_3d = res['gt_bboxes_3d']
        gt_labels_3d = res['gt_labels_3d']

        points = points.data
        self.voxelize(points, [0.05, 0.05, 0.1])
        exit(0)


        return points


if __name__ == "__main__":
    from mmcv import Config
    #cfg = Config.fromfile("config/dataset/kitti-3d-car.py")
    # cfg = Config.fromfile("config/dataset/kitti-3d-3class.py")
    cfg = Config.fromfile("./configs/_base_/datasets/kitti-3d-3class.py")
    dataset_cfg = cfg.data.train
    dataset = Dataset(dataset_cfg, [[0.05, 0.05, 0.1]], [0, -40, -3, 70.4, 40, 1])
    print("len", len(dataset))
    for idx in range(len(dataset)):
        datas = dataset.__getitem__(idx)
        print("points shape", datas.shape)
        if idx==3: break
