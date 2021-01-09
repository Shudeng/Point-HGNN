import torch
from torch import nn
#from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
#from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from torch_scatter import scatter_max
from collections import OrderedDict
import torch.distributed as dist

from construct_graph import voxelize, inter_level_graph, intra_level_graph


def multi_layer_neural_network_fn(Ks):
    linears = []
    for i in range(1, len(Ks)):
        linears += [
            nn.Linear(Ks[i - 1], Ks[i]),
            nn.ReLU(),
            # nn.BatchNorm1d(Ks[i])
        ]
    return nn.Sequential(*linears)


def max_aggregation_fn(features, index, l):
    """
    args:
        features: N x dim
        index: N x 1, e.g.  [0,0,0,1,1,...l,l]
        l: lenght of keypoints

    return:
        set_features: l x dim
    """
    index = index.unsqueeze(-1).expand(-1, features.shape[-1])
    index = index.to(features.device)  # N x dim
    set_features = torch.zeros((l, features.shape[-1]), device=features.device).permute(1, 0).contiguous()  # len x dim
    output = scatter_max(features.permute(1, 0).contiguous(), index.permute(1, 0).contiguous(), out=set_features)
    set_features, argmax= output
    set_features = set_features.permute(1, 0)
    return set_features


class BasicBlock(nn.Module):
    def __init__(self, in_inter_channels, out_inter_channels):
        """
        args:
            in_inter_channels: inter channels of in_linear network
            out_inter_channels: inter channels of out_linear network

        """
        super(BasicBlock, self).__init__()
        assert in_inter_channels[-1] == out_inter_channels[0]

        # self.inplanes = inplanes
        # self.outplanes = outplanes
        self.in_linear = multi_layer_neural_network_fn(in_inter_channels)
        self.out_linear = multi_layer_neural_network_fn(out_inter_channels)

    def forward(self, last_coors, last_features, current_coors, edge):
        """
        args:
            last_coors: coordinates of last level, N x 3
            last_features: features of last level, N x f
            current_coors: coordinates of current level, M x 3
            edge: edge, 2 x E, [current_index, last_index]

        return:
            current_features, M x outplanes
        """
        current_indices = edge[0, :]
        last_indices = edge[1, :]

        center_coors = current_coors[current_indices]  # E x 3
        neighbor_coors = last_coors[last_indices]  # E x 3
        neighbor_features = last_features[last_indices]  # E x f
        neighbor_features = torch.cat([neighbor_features, neighbor_coors - center_coors], dim=1)  # E x (3+f)
        neighbor_features = self.in_linear(neighbor_features)

        current_features = max_aggregation_fn(neighbor_features, current_indices, len(current_coors))
        return self.out_linear(current_features)


class DownsampleBlock(nn.Module):
    def __init__(self, in_inter_channels, out_inter_channels):
        super(DownsampleBlock, self).__init__()
        self.basic_block = BasicBlock(in_inter_channels, out_inter_channels)

    def forward(self, last_coors, last_features, current_coors, edge):
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
        update_features = self.basic_block(coors, features, coors, edge)  # N x f, can be changed to attention mode
        assert update_features.shape[1] == features.shape[1]

        return self.after_cat_linear(features + update_features)


class UpsampleBlock(nn.Module):
    def __init__(self, in_inter_channels, out_inter_channels, before_cat_inter_channels, after_cat_inter_channels):
        super(UpsampleBlock, self).__init__()
        self.basic_block = BasicBlock(in_inter_channels, out_inter_channels)
        self.before_cat_linear = multi_layer_neural_network_fn(before_cat_inter_channels)
        self.after_cat_linear = multi_layer_neural_network_fn(after_cat_inter_channels)

    def forward(self, current_coors, current_features, last_coors, last_features, edge):
        # last corresponds to the upsampled point?
        update_features = self.basic_block(current_coors, current_features, last_coors,
                                           edge)  # can be changed to attention mode

        before_cat_features = self.before_cat_linear(last_features)
        after_cat_features = before_cat_features + update_features
        return self.after_cat_linear(after_cat_features)


class HGNN(nn.Module):
    def __init__(self, downsample_voxel_sizes, inter_radius, intra_radius,
                 max_num_neighbors):
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
        self.max_num_neighbors = max_num_neighbors
        #self.num_classes = num_classes
        #self.head_type = head_type
        #self.box_encoding_len = box_encoding_len
        #self.cfg = cfg
        #self.train_cfg = cfg['train_cfg']
        #self.test_cfg = cfg['test_cfg']

        # self.linear = nn.Linear(10, 100)
        self.downsample1 = DownsampleBlock(in_inter_channels=(4+3, 32, 64), out_inter_channels=(64, 64))
        self.graph1 = GraphBlock(in_inter_channels=(64 + 3, 64), out_inter_channels=(64, 64),
                                   after_cat_inter_channels=(64, 64))
        self.downsample2 = DownsampleBlock((64 + 3, 128), (128, 128))
        self.graph2 = GraphBlock((128 + 3, 128), (128, 128), (128, 128))
        self.downsample3 = DownsampleBlock((128 + 3, 300), (300, 300))
        self.graph3 = GraphBlock((300 + 3, 300), (300, 300), (300, 300))
        self.upsample1 = UpsampleBlock(in_inter_channels=(303, 128), out_inter_channels=(128, 128),
                                       before_cat_inter_channels=(128, 128), after_cat_inter_channels=(128, 128))
        self.graph2_update = GraphBlock((128 + 3, 128), (128, 128), (128, 128))
        self.upsample2 = UpsampleBlock((128 + 3, 64), (64, 64), (64, 64), (64, 64))
        self.graph1_update = GraphBlock((64 + 3, 64), (64, 64), (64, 64))
        self.upsample3 = UpsampleBlock((64 + 3, 32), (32, 16), (4, 16), (16, 4))  # not utilized

    def get_levels_coordinates(self, point_coordinates, voxel_sizes):
        """
        args:
            point_coordinates: a tensor, N x 3
            voxel_sizes: a list of list, its length is 3 defaut;
                example: [[0.05,0.05,0.1], [0.07 , 0.07, 0.12], [0.09, 0.09, 0.14]]
        return:
            downsample_point_coordinates: a list of tensor, whose length is the same as voxel.

        """
        l1_point_coordinates, l1_point_indices = voxelize(point_coordinates, voxel_sizes[0])
        l2_point_coordinates, l2_point_indices = voxelize(point_coordinates, voxel_sizes[1])
        l3_point_coordinates, l3_point_indices = voxelize(point_coordinates, voxel_sizes[2])

        return [l1_point_coordinates, l2_point_coordinates, l3_point_coordinates], \
                [l1_point_indices, l2_point_indices, l3_point_indices]

    def forward(self,
                points,
                img_metas=None,
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                mode='train'):
        """args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.

        """

        #assert len(points) == 1 and len(img_metas) == 1 and len(gt_bboxes_3d) == 1 and len(gt_labels_3d) == 1
        assert len(points)==1
        #points = points[0][:100, :]
        points = points[0]

        ## step 1: construct graph
        coordinates, indices = self.get_levels_coordinates(points[:, :3], self.downsample_voxel_sizes)
        coordinates = [points[:, :3]] + coordinates

        # coordinates = [points[:, :3]] + self.get_levels_coordinates(points[:, :3], self.downsample_voxel_sizes)
        inter_graphs = {}
        intra_graphs = {}
        for i in range(len(coordinates)):
            if i != len(coordinates) - 1:
                inter_graphs["{}_{}".format(i, i + 1)], inter_graphs["{}_{}".format(i + 1, i)] = \
                    inter_level_graph(coordinates[i], coordinates[i + 1], self.inter_radius[i],
                                      max_num_neighbors=self.max_num_neighbors)
            if i != 0:
                # construct intra graph
                intra_graphs["{}_{}".format(i, i)] = intra_level_graph(coordinates[i], self.intra_radius[i - 1])

        for i, coordinate in enumerate(coordinates):
            print("coordinate ", i, coordinate.shape)
        for k, v in inter_graphs.items():
            print(k, v.shape)
        for k, v in intra_graphs.items():
            print(k, v.shape)
        # print(inter_graphs["2_3"][:, :10])

        ## step 2: extract features (downsample and upsample with hierarchical connect) via graph

        p1 = self.downsample1(last_coors=coordinates[0], last_features=points,
                              current_coors=coordinates[1], edge=inter_graphs["0_1"])
        encode_p1 = self.graph1(coors=coordinates[1], features=p1, edge=intra_graphs["1_1"])
        p2 = self.downsample2(coordinates[1], encode_p1, coordinates[2], inter_graphs["1_2"])
        encode_p2 = self.graph2(coordinates[2], p2, intra_graphs["2_2"])
        p3 = self.downsample3(coordinates[2], encode_p2, coordinates[3], inter_graphs["2_3"])
        p3 = self.graph3(coordinates[3], p3, intra_graphs["3_3"])
        decode_p2 = self.upsample1(current_coors=coordinates[3], current_features=p3,
                                   last_coors=coordinates[2], last_features=encode_p2, edge=inter_graphs["3_2"])
        p2 = self.graph2_update(coordinates[2], decode_p2, intra_graphs["2_2"])
        decode_p1 = self.upsample2(coordinates[2], p2, coordinates[1], encode_p1, inter_graphs["2_1"])
        p1 = self.graph1_update(coordinates[1], decode_p1, intra_graphs["1_1"])
        # p0 = self.upsample3(coordinates[1], p1, coordinates[0], points, inter_graphs["1_0"])

        print('size of extracted point features: ', p1.size())  # the first downsample graph
        #point_features = p1
        # (logits, box_encodings) = self.predictor(point_features)
        #results = self.predictor(point_features)
        #print("results.shape", results[0].shape)

        ## step 3: feed features to classify and regress box via head
        indices_1, indices_2, indices_3 = indices 
        for idc in indices:
            print(idc.size())

        # since it's one batch now, we need to unsqueeze one dimension for the inputs of VoteHead.
        # fp_xyz: Layer x Batch x N x 3; fp_features: L x B x f x N; fp_indices: L x B x N.


        fp_xyz = [coordinates[3].unsqueeze(0), 
                  coordinates[2].unsqueeze(0), 
                  coordinates[1].unsqueeze(0)]
        fp_features = [p3.unsqueeze(0).permute(0, 2, 1), 
                       p2.unsqueeze(0).permute(0, 2, 1), 
                       p1.unsqueeze(0).permute(0, 2, 1)]
        fp_indices = [indices_3.unsqueeze(0), 
                      indices_2.unsqueeze(0), 
                      indices_1.unsqueeze(0)]
        feat_dict = {'fp_xyz': fp_xyz,
                    'fp_features': fp_features,
                    'fp_indices': fp_indices,
        }
        return feat_dict

