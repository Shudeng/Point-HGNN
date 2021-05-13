import torch
from torch import nn
#from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
#from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from torch_scatter import scatter_max, scatter
from collections import OrderedDict
import torch.distributed as dist

from construct_graph import voxelize, inter_level_graph, intra_level_graph


def multi_layer_neural_network_fn(Ks):
    linears = []
    for i in range(1, len(Ks)):
        linears += [
            nn.Linear(Ks[i - 1], Ks[i]),
            nn.ReLU(),
            nn.BatchNorm1d(Ks[i])
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


    output = scatter(features.contiguous(), index.contiguous(), dim=0, dim_size=l, reduce="max")
    #set_features, argmax= output
    set_features = output
    return set_features

def small_to_large(features_list, index_list, length_list):
    # concat several small graphs to a large graph
    new_index_list = [index_list[0]]
    for i in range(1, len(index_list)): new_index_list += [index_list[i]+length_list[i-1]]
    features_cat = torch.cat(features_list, dim=0)
    index_cat = torch.cat(index_list, dim=0)

    return features_cat, index_cat

def large_to_small(large_graph_features, length_list):
    output = large_graph_features

    new_graphs = []
    start = 0
    for i in range(len(length_list)):
        end = start + length_list[i]
        new_graphs += [output[start:end]]
        start = end
    return new_graphs


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
            last_coors: list of corrdinates of last level
            last_features: list of features of last level
            current_coors: list of coordinates of current level
            edge: list of edges

        return:
            update_features, a tensor
            length_list: length of graph of current level
        """
        features_list, indices_list, length_list = [], [], []
        total_len = 0
        for i in range(len(edge)):
            current_indices = edge[i][0, :].long()
            last_indices = edge[i][1, :].long()

            """
            print("current_indices.max()", current_indices.max())
            print("last_indices.max()", last_indices.max())
            """

            center_coors = current_coors[i][current_indices]  # E x 3
            neighbor_coors = last_coors[i][last_indices]  # E x 3
            neighbor_features = last_features[i][last_indices]  # E x f
            neighbor_features = torch.cat([neighbor_features, neighbor_coors - center_coors], dim=1)  # E x (3+f)

            features_list += [neighbor_features]
            indices_list += [current_indices]
            length_list += [len(current_coors[i])]
            total_len += len(current_coors[i])

        features, indices = small_to_large(features_list, indices_list, length_list)
        features = self.in_linear(features)
        """
        print("features.shape", features.shape)
        print("indices", indices.shape, indices.max(), indices.min())
        print("total_len", total_len)
        print("length_list", length_list)
        """

        current_features = max_aggregation_fn(features, indices, total_len)

        return self.out_linear(current_features), length_list


class DownsampleBlock(nn.Module):
    def __init__(self, in_inter_channels, out_inter_channels):
        super(DownsampleBlock, self).__init__()
        self.basic_block = BasicBlock(in_inter_channels, out_inter_channels)

    def forward(self, last_coors, last_features, current_coors, edge):
        features, length_list = self.basic_block(last_coors, last_features, current_coors, edge)
        features_list = large_to_small(features, length_list)

        return features_list


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
        update_features, length_list = self.basic_block(coors, features, coors, edge)  # N x f, can be changed to attention mode
        features = torch.cat(features, dim=0)
        assert update_features.shape[1] == features.shape[1]
        update_features = self.after_cat_linear(features + update_features)
        features_list = large_to_small(update_features, length_list)

        return features_list


class UpsampleBlock(nn.Module):
    def __init__(self, in_inter_channels, out_inter_channels, before_cat_inter_channels, after_cat_inter_channels):
        super(UpsampleBlock, self).__init__()
        self.basic_block = BasicBlock(in_inter_channels, out_inter_channels)
        self.before_cat_linear = multi_layer_neural_network_fn(before_cat_inter_channels)
        self.after_cat_linear = multi_layer_neural_network_fn(after_cat_inter_channels)

    def forward(self, current_coors, current_features, last_coors, last_features, edge):
        # last corresponds to the upsampled point?
        update_features, length_list = self.basic_block(current_coors, current_features, last_coors,
                                           edge)  # can be changed to attention mode

        last_features = torch.cat(last_features, dim=0)
        before_cat_features = self.before_cat_linear(last_features)
        after_cat_features = before_cat_features + update_features
        features_list = large_to_small(after_cat_features, length_list)
        return features_list

class HGNN(nn.Module):
    def __init__(self, downsample_voxel_sizes, inter_radius, intra_radius,
                 max_num_neighbors):
        """
        args:
            downsample_voxel_sizes: a list of list, its length is 3 defaut;
            example: [[0.05,0.05,0.1], [0.07 , 0.07, 0.12], [0.09, 0.09, 0.14]]
            inter_radius: a list, the radius for constructing graphs.
            intra_radius: a list, the radius for constructing graphs
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
        self.downsample1 = DownsampleBlock(in_inter_channels=(4+3, 32, 64, 128, 300), out_inter_channels=(300, 300))

        self.graph1s = nn.ModuleList()
        for _ in range(3):
            self.graph1s.append(GraphBlock(in_inter_channels=(300 + 3, 300), out_inter_channels=(300, 300), 
                                   after_cat_inter_channels=(300, 300)) )
        self.downsample2 = DownsampleBlock((300 + 3, 300), (300, 300))

        self.graph2s = nn.ModuleList()
        for _ in range(3):
            self.graph2s.append( GraphBlock((300 + 3, 300), (300, 300), (300, 300)) )


        self.downsample3 = DownsampleBlock((300 + 3, 300), (300, 300))

        self.graph3s = nn.ModuleList()
        for _ in range(3):
            self.graph3s.append( GraphBlock((300 + 3, 300), (300, 300), (300, 300)))

        self.upsample1 = UpsampleBlock(in_inter_channels=(303, 300), out_inter_channels=(300, 300),
                                       before_cat_inter_channels=(300, 300), after_cat_inter_channels=(300,300))


        self.graph2_updates = nn.ModuleList()
        for _ in range(3):
            self.graph2_updates.append( GraphBlock((300 + 3, 300), (300, 300), (300, 300)) )


        self.upsample2 = UpsampleBlock((300 + 3, 300), (300,  300), (300, 300), ( 300,300))
        self.graph1_updates = nn.ModuleList()
        for _ in range(3):
            self.graph1_updates.append(  GraphBlock((300 + 3, 300), (300,300), (300, 300)) )


        #self.upsample3 = UpsampleBlock((300 + 3, 32), (32, 16), (4, 16), (16, 4))  # not utilized

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
                mode='train', 
                **kwargs):
        """args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.

        """

        #assert len(points) == 1 and len(img_metas) == 1 and len(gt_bboxes_3d) == 1 and len(gt_labels_3d) == 1
        #assert len(points)==1

        coordinates = [kwargs["keypoints_{}".format(level)] for level in range(4)]
        indices = [kwargs["indices_{}".format(level)] for level in range(1,4)]
        graphs = kwargs


        ## step 2: extract features (downsample and upsample with hierarchical connect) via graph

        p1 = self.downsample1(last_coors=coordinates[0], last_features=points,
                              current_coors=coordinates[1], edge=graphs["graph_0_1"])

        edge = graphs["graph_0_1"][0]
        encode_p1 = p1

        for i in range(len(self.graph1s)):
            encode_p1 = self.graph1s[i](coors=coordinates[1], features=encode_p1, edge=graphs["graph_1_1"])

        p2 = self.downsample2(coordinates[1], encode_p1, coordinates[2], graphs["graph_1_2"])

        encode_p2 = p2
        for i in range(len(self.graph2s)):
            encode_p2 = self.graph2s[i](coordinates[2], encode_p2, graphs["graph_2_2"])

        p3 = self.downsample3(coordinates[2], encode_p2, coordinates[3], graphs["graph_2_3"])

        for i in range(len(self.graph3s)):
            p3 = self.graph3s[i](coordinates[3], p3, graphs["graph_3_3"])


        decode_p2 = self.upsample1(current_coors=coordinates[3], current_features=p3,
                                   last_coors=coordinates[2], last_features=encode_p2, edge=graphs["graph_3_2"])

        p2 = decode_p2

        for i in range(len(self.graph2_updates)):
            p2 = self.graph2_updates[i](coordinates[2], p2, graphs["graph_2_2"])

        decode_p1 = self.upsample2(coordinates[2], p2, coordinates[1], encode_p1, graphs["graph_2_1"])
        p1 = decode_p1
        for i in range(len(self.graph1_updates)):
            p1 = self.graph1_updates[i](coordinates[1], p1, graphs["graph_1_1"])
        # p0 = self.upsample3(coordinates[1], p1, coordinates[0], points, graphs["1_0"])

        indices_1, indices_2, indices_3 = indices 
        max_nums = []
        for indices_i in indices:
            max_num = -1e10
            for indice in indices_i:
                if max_num < len(indice): max_num = len(indice)
            max_nums += [max_num]

        # since it's one batch now, we need to unsqueeze one dimension for the inputs of VoteHead.
        # fp_xyz: Layer x Batch x N x 3; fp_features: L x B x f x N; fp_indices: L x B x N.

        """
        for k, coors in enumerate(coordinates):
            for l, coor in enumerate(coors):
                print(k, l, coor.shape)
        print("*"*20+"indices"+"*"*10)
        for k, inds in enumerate(indices):
            for l, ind in enumerate(inds):
                print(k,l, ind.shape)
        for l, point in enumerate(p1):
            print(3, l, point.shape)
        for l, point in enumerate(p2):
            print(3, l, point.shape)
        for l, point in enumerate(p3):
            print(3, l, point.shape)
        """



        def padlist2tensor(listoftensors, max_num, pad_value):
            for i in range(len(listoftensors)):
                if pad_value != -1:
                    listoftensors[i] = torch.cat([
                                listoftensors[i], 
                                listoftensors[i].new_ones(max_num-len(listoftensors[i]), *listoftensors[i].shape[1:])*pad_value
                            ]).unsqueeze(0)
                else: # pad indices
                    print("listoftensors[i]", listoftensors[i].shape)
                    print(listoftensors[i])
                    listoftensors[i] = torch.cat([
                                listoftensors[i], 
                                torch.arange(len(listoftensors[i]), max_num).to(listoftensors[i].device)
                            ]).unsqueeze(0)
                    print(listoftensors[i])


            return torch.cat(listoftensors, dim=0)



        fp_xyz = [
                padlist2tensor(coordinates[3], max_nums[2], 1e10),
                padlist2tensor(coordinates[2], max_nums[1], 1e10),
                padlist2tensor(coordinates[1], max_nums[0], 1e10)]

        #print("padlist2tensor(p3, max_nums[2], 0).permute(0, 2, 1).shape",padlist2tensor(p3, max_nums[2], 0).permute(0, 2, 1).shape)
        fp_features = [
                padlist2tensor(p3, max_nums[2], 0).permute(0, 2, 1),
                padlist2tensor(p2, max_nums[1], 0).permute(0, 2, 1),
                padlist2tensor(p1, max_nums[0], 0).permute(0, 2, 1),
                ]

        fp_indices = [
                padlist2tensor(indices_3, max_nums[2], 0), 
                padlist2tensor(indices_2, max_nums[1], 0), 
                padlist2tensor(indices_1, max_nums[0], 0)]

        feat_dict = {'fp_xyz': fp_xyz,
                'fp_features': fp_features,
                'fp_indices': fp_indices,
        }
        return feat_dict

