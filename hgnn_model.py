import torch
from torch import nn
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.models.detectors import Base3DDetector
from construct_graph import voxelize, inter_level_graph, intra_level_graph

class HGNN(nn.Module):
    def __init__(self, downsample_voxel_sizes, inter_radius, intra_radius):
        """
        args:
            downsample_voxel_sizes: a list of list, its length is 3 defaut;
            example: [[0.05,0.05,0.1], [0.07 , 0.07, 0.12], [0.09, 0.09, 0.14]]
            inter_radius: a list, the radius for constructing inter_graphs.
            intra_radius: a list, the radius for constructing intra_graphs
        """

        super(HGNN, self).__init__()
        self.downsample_voxel_sizes = downsample_voxel_sizes
        self.inter_radius = inter_radius
        self.intra_radius = intra_radius

        self.linear = nn.Linear(10, 100)

    def get_levels_coordinates(self, point_coordinates, voxel_sizes):
        """
        args:
            point_coordinates: a tensor, N x 3
            voxel_sizes: a list of list, its length is 3 defaut; 
                example: [[0.05,0.05,0.1], [0.07 , 0.07, 0.12], [0.09, 0.09, 0.14]]
        return:
            downsample_point_coordinates: a list of tensor, whose length is the same as voxel.

        """
        l1_point_coordinates = voxelize(point_coordinates, voxel_sizes[0])
        l2_point_coordinates = voxelize(point_coordinates, voxel_sizes[1])
        l3_point_coordinates = voxelize(point_coordinates, voxel_sizes[2])

        return [l1_point_coordinates, l2_point_coordinates, l3_point_coordinates]

    def 

    def forward(self, 
            points, 
            img_metas, 
            gt_bboxes_3d, 
            gt_labels_3d):
        """args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.

        """
        # print("len(points)", len(points))
        # print("len(img_metas)", len(img_metas))
        # print("len(gt_bboxes_3d)", len(gt_bboxes_3d))
        # print("len(gt_labels_3d)", len(gt_labels_3d))


        # Note: currently we only support one batch for single gpu. 
        #       multi-batch for single gpu need further work.

        assert len(points)==1 and len(img_metas)==1 and len(gt_bboxes_3d)==1 and len(gt_labels_3d)==1
        points = points[0]

        ## step 1: construct graph
        coordinates = [points[:, :3]] + self.get_levels_coordinates(points[:, :3], self.downsample_voxel_sizes)
        inter_graphs = {}
        intra_graphs = {}
        for i in range(len(coordinates)):
            if i!= len(coordinates)-1:
                inter_graphs["{}_{}".format(i, i+1)], inter_graphs["{}_{}".format(i+1, i)] = \
                    inter_level_graph(coordinates[i], coordinates[i+1], self.inter_radius[i])
            if i!=0:
                # construct intra graph
                intra_graphs["{}_{}".format(i,i)] = intra_level_graph(coordinates[i], self.intra_radius[i-1])

        """
        for i, coordinate in enumerate(coordinates):
            print("coordinate ", i, coordinate.shape)
        for k, v in inter_graphs.items():
            print(k, v.shape)
        for k, v in intra_graphs.items():
            print(k, v.shape)
        """








        ## step 2 

        return x

    def forward_train(self, x):
        pass


    def aug_test(self, x):
        pass


    def simple_test(self, x):
        pass
