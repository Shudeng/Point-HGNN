import os
import time
import torch
from mmdet3d.datasets import build_dataset
from mmdet3d.ops.voxel.voxel_layer import dynamic_voxelize
from mmcv.parallel import DataContainer as DC
from torch_cluster import radius, radius_graph
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset_cfg):
        self.dataset = build_dataset(dataset_cfg)
        self.CLASSES = self.dataset.CLASSES
        self.flag = self.dataset.flag


        self.max_num_neighbors = 32
        self.downsample_voxel_sizes = [[0.5, 0.5, 0.5], [0.8, 0.8, 0.8], [1.1, 1.1, 1.1]]
        self.inter_radius = [1.0, 1.5, 2.0]
        self.intra_radius = [1.5, 2.0, 2.5]

    def voxelize(self, points, voxel_size):
        voxel_size = torch.tensor(voxel_size).to(points.device)
        rescale_points = points / voxel_size
        rescale_points = rescale_points.long()

        keypoints = set()
        indices = []
        for i, point in enumerate(rescale_points):
            point = tuple(point.tolist())
            if not point in keypoints:
                keypoints.add(point)
                indices += [i]

        return torch.tensor(list(keypoints)), torch.tensor(indices).long()

    def inter_level_graph(self, points: torch.Tensor, key_points: torch.Tensor, radiu, max_num_neighbors=32):
        """
        args: 
            points: N x 3
            key_points: M x 3
        return:
            downsample_graph: E_1 x 2, [center_node, neighbor_node], E_1 is the number of edges
            upsample_graph: E_1 x 2 [neighbor_node, center_node]
        """
        batch_x = torch.tensor([0]*len(points)).to(points.device)
        batch_y = torch.tensor([0]*len(key_points)).to(key_points.device)

        downsample_graph = radius(points, key_points, radiu, batch_x, batch_y, max_num_neighbors=max_num_neighbors)
        # upsample_graph = radius(key_points, points, radiu, batch_y, batch_x, max_num_neighbors=max_num_neighbors)
        #upsample_graph = downsample_graph[[1, 0], :]
        #return downsample_graph.to(points.device), upsample_graph.to(points.device)

        return downsample_graph.to(points.device)

    def intra_level_graph(self, key_points: torch.Tensor, radiu, loop: bool=False):
        """
        args:
            key_points: nodes of specific level
            loop: True if node has edge to itself.
        return: self_level_graph E x 2, [center_node, neighbor_node]
        """
        batch_x = torch.tensor([0]*len(key_points)).to(key_points.device)
        intra_graph = radius_graph(key_points, radiu, batch_x, loop)
        return intra_graph




    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        res = self.dataset.__getitem__(idx)
        points = res['points']
        points = points.data.cuda()
        coordinates = [points[:, :3]]
        res["keypoints_{}".format(0)] = DC(points.data[:,:3])


        ## voxelize
        for level in range(3):
            keypoints, indices = self.voxelize(points.data[:,:3], self.downsample_voxel_sizes[level])
            keypoints, indices = keypoints.float().cuda(), indices.cuda()
            res["keypoints_{}".format(level+1)] = DC(keypoints)
            res["indices_{}".format(level+1)] = DC(indices)
            coordinates += [keypoints]

        for i in range(len(coordinates)):
            if i != len(coordinates) - 1:
                graph = self.inter_level_graph(coordinates[i], coordinates[i + 1], self.inter_radius[i],
                        max_num_neighbors=self.max_num_neighbors)
                res["graph_{}_{}".format(i, i+1)] = DC(graph)
                res["graph_{}_{}".format(i+1, i)] = DC(graph[[1,0], :])

            if i!=0:
                graph = self.intra_level_graph(coordinates[i], self.intra_radius[i - 1])
                res["graph_{}_{}".format(i, i)] = DC(graph)

        return res


if __name__ == "__main__":
    from mmcv import Config
    #cfg = Config.fromfile("config/dataset/kitti-3d-car.py")
    # cfg = Config.fromfile("config/dataset/kitti-3d-3class.py")
    cfg = Config.fromfile("configs/_base_/datasets/sunrgbd-3d-10class.py")
    dataset_cfg = cfg.data.train
    dataset = MyDataset(dataset_cfg)
    print("len", len(dataset))
    for idx in range(len(dataset)):
        start = time.time()
        datas = dataset.__getitem__(idx)
        print("datas", datas)
        print(datas.keys())
        print("points shape", datas['points'].data.shape, "time:", time.time()-start)

        if idx==3: break
