import torch
from torch import nn
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.models.detectors import Base3DDetector
from construct_graph import voxelize, inter_level_graph, intra_level_graph

def multi_layer_neural_network_fn(Ks):
    linears = []
    for i in range(1, len(Ks)):
        linears += [
        nn.Linear(Ks[i-1], Ks[i]),
        nn.ReLU(),
        nn.BatchNorm1d(Ks[i])]
    return nn.Sequential(*linears)

def max_aggregation_fn(features, index, l):
    """
    Arg: features: N x dim
    index: N x 1, e.g.  [0,0,0,1,1,...l,l]
    l: lenght of keypoints
    """
    index = index.unsqueeze(-1).expand(-1, features.shape[-1]) # N x 64
    set_features = torch.zeros((l, features.shape[-1]), device=features.device).permute(1,0).contiguous() # len x 64
    set_features, argmax = scatter_max(features.permute(1,0), index.permute(1,0), out=set_features)
    set_features = set_features.permute(1,0)
    return set_features

class BasicBlock(nn.Module):
    def __init__(self, in_inter_channels, out_inter_channels):
        """
        args:
            in_inter_channels: inter channels of in_linear network
            out_int_channels: inter channels of out_linear network

        """
        super(DownsampleBlock, self).__init__()
        assert in_inter_channels[-1] == out_inter_channels[0]

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.in_linear = multi_layer_neural_network_fn(in_inter_channels)
        self.out_linear = multi_layer_neural_network_fn(out_inter_channels)

    def forward(self, last_coors, last_features, current_coors, edge):
        """
        args:
            last_coors: coordinates of last level, N x 3
            last_features: features of last level, N x f
            current_coors: coordinates of current level, M x 3
            edge: edge, 2 x E, [current_index, last_index]

        return: current_features, M x outplanes
        """
        current_indices = edge[0, :]
        last_indices = edge[1, :]

        center_coors = current_coors[current_indices] # E x 3
        neighbor_coors = last_coors[last_indices] # E x 3
        neighbor_features = last_features[last_indices] # E x f
        neighbor_features = torch.cat([neighbor_features, neighbor_coors-center_coors], dim=1) # E x (3+f)
        neighbor_features = self.linear(neighbor_features)

        current_features = max_aggregation_fn(neighbor_features, current_indices, len(current_coors))
        return self.out_linear(current_features)

class DownsampleBlock(nn.Module):
    def __init__(self, in_inter_channels, out_inter_channels):
        super(DownsampleBlock, self).__init__()
        self.basic_block = BasicBlock(in_inter_channels, out_inter_channels)

    def forward(self, last_features, current_coors, edge):
        return self.basic_block(last_coors, last_features, current_coors, edge)

class GraphBlock(nn.Module):
    def __init__(self, in_inter_channels, out_inter_channels, after_cat_inter_channels):
        super(GraphBlock, self).__init__()
        self.basic_block = BasicBlock(in_inter_channels, out_inter_channels)
        self.after_cat_linear = multi_layer_neural_network_fn(after_cat_inter_channels)

    def forward(self, coors, features, edge):
        """
        args:
            coors: coordinates of current level, N x 3
            features: features of current level, N x f
            edge: edge of intra graph

        return: updated features
        """
        update_features = self.basic_block(coors, features, coors, edge) # N x f, can be changed to attention mode
        assert update_features.shape[1] == features.shape[1]

        return self.after_cat_linear(features + update_features)

class UpsampleBlock(nn.Module):
    def __init__(self, in_inter_channels, out_inter_channels, before_cat_inter_channels, after_cat_inter_channels):
        super(UpsampleBlock, self).__init__()
        self.basic_block = BasicBlock(in_inter_channels, out_inter_channels)
        self.before_cat_linear = multi_layer_neural_network_fn(before_cat_inter_channels)
        self.after_cat_linear = multi_layer_neural_network_fn(after_cat_inter_channels)

    def forward(self, current_coors, current_features, last_coors, last_features, edge):
        update_features = self.basic_block(current_coors, current_features, last_coors) # can be changed to attention mode
        
        before_cat_features = self.before_cat_linear(last_features)
        after_cat_features = before_cat_features + update_features
        return self.after_cat_linear(after_cat_features)

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
