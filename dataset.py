import os
import time
import torch
from mmdet3d.datasets import build_dataset, SUNRGBDDataset
from mmdet3d.ops.voxel.voxel_layer import dynamic_voxelize
from mmcv.parallel import DataContainer as DC
from torch_cluster import radius, radius_graph
from torch.utils.data import Dataset
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class MyDataset(SUNRGBDDataset):
    def __init__(self, dataset_cfg, dict_test_mode=None):
        if dict_test_mode is None:
            self.dataset = build_dataset(dataset_cfg)
        else:
            self.dataset = build_dataset(dataset_cfg, dict_test_mode)

        self.CLASSES = self.dataset.CLASSES
        if dict_test_mode is not None:
            self.test_mode = dict_test_mode['test_mode']
            self.data_infos = self.dataset.data_infos
            self.box_type_3d = self.dataset.box_type_3d
            self.box_mode_3d = self.dataset.box_mode_3d
            self.cat2id = self.dataset.cat2id
            self.filter_empty_gt = self.dataset.filter_empty_gt

        else:self.test_mode = False

        try:
            self.flag = self.dataset.flag 
        except: pass



        self.max_num_neighbors = 256
        self.downsample_voxel_sizes = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]
        self.inter_radius = [0.3, 0.5, 0.7]
        self.intra_radius = [0.4, 0.6, 0.8]

        self.device="cpu"
        """
        self.downsample_voxel_sizes = [[0.5, 0.5, 0.5], [0.8, 0.8, 0.8], [1.1, 1.1, 1.1]]
        self.inter_radius = [1.0, 1.5, 2.0]
        self.intra_radius = [1.5, 2.0, 2.5]
        """

    def voxelize(self, points, voxel_size):
        voxel_size = torch.tensor(voxel_size)
        rescale_points = points / voxel_size.to(points.device)
        rescale_points = rescale_points.long()

        keypoints = set()
        indices = []
        for i, point in enumerate(rescale_points):
            point = tuple(point.tolist())
            if not point in keypoints:
                keypoints.add(point)
                indices += [i]

        return torch.tensor(list(keypoints))*voxel_size, torch.tensor(indices).long()

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
        intra_graph = radius_graph(key_points, radiu, batch_x, loop, max_num_neighbors=self.max_num_neighbors)
        return intra_graph




    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        res = self.dataset.__getitem__(idx)
        #return res
        points = res['points']

        if self.test_mode:
            points = points[0]
        points = points.data.to(self.device)
        coordinates = [points[:, :3]]
        res["keypoints_{}".format(0)] = DC(points.data[:,:3])


        ## voxelize
        for level in range(3):
            keypoints, indices = self.voxelize(points.data[:,:3], self.downsample_voxel_sizes[level])
            keypoints, indices = keypoints.float().to(self.device), indices.to(self.device)
            res["keypoints_{}".format(level+1)] = DC(keypoints)
            res["indices_{}".format(level+1)] = DC(indices)
            #print("keypoints.shape", keypoints.shape)

            coordinates += [keypoints]

        for i in range(len(coordinates)):
            if i != len(coordinates) - 1:

                assert len(coordinates[i]) != 1
                assert len(coordinates[i+1]) != 1

                graph = self.inter_level_graph(coordinates[i], coordinates[i + 1], self.inter_radius[i],
                        max_num_neighbors=self.max_num_neighbors)

                assert graph.shape[1]!=0
                assert graph[0,:].max() < len(coordinates[i+1])
                assert graph[1,:].max() < len(coordinates[i])
                #print("in_graph.shape", graph.shape)

                res["graph_{}_{}".format(i, i+1)] = DC(graph)
                res["graph_{}_{}".format(i+1, i)] = DC(graph[[1,0], :])

            if i!=0:
                #print("len(coordinates[i])", len(coordinates[i]))

                graph = self.intra_level_graph(coordinates[i], self.intra_radius[i - 1])
                assert graph[0, :].max() < len(coordinates[i])
                assert graph[1, :].max() < len(coordinates[i])

                #print("graph.shape", graph.shape)
                res["graph_{}_{}".format(i, i)] = DC(graph)

        return res


if __name__ == "__main__":
    from mmcv import Config
    #cfg = Config.fromfile("config/dataset/kitti-3d-car.py")
    # cfg = Config.fromfile("config/dataset/kitti-3d-3class.py")
    cfg = Config.fromfile("configs/_base_/datasets/sunrgbd-3d-10class.py")
    #dataset_cfg = cfg.data.train
    #dataset = MyDataset(dataset_cfg)

    dataset_cfg = cfg.data.val
    dataset = MyDataset(dataset_cfg, dict(test_mode=True))
    for idx in range(len(dataset)):
        start = time.time()
        datas = dataset.__getitem__(idx)
        #print("points shape", datas['points'].data.shape, "time:", time.time()-start)

        if idx==3: break
