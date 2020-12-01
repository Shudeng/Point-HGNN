import torch
from torch_cluster import radius, radius_graph

def voxelize(points: torch.Tensor, voxel_size):
    """
    args: 
        points: torch.Tensor and its shape is N x 3
        voxel_size: a tuple or a list, [x_size, y_size, z_size]

    return: voxelize coordinates
    """
    print("points shape", points.shape)

    voxel_size = torch.tensor(voxel_size).to(points.device)
    key_points = points / voxel_size
    key_points = key_points.long()
    key_points = torch.unique(key_points, dim=0)
    return key_points.float()

def inter_level_graph(points: torch.Tensor, key_points: torch.Tensor, radiu):
    """
    args: 
        points: N x 3
        key_points: M x 3
    return:
        downsample_graph: E_1 x 2, [center_node, neighbor_node], E_1 is the number of edges
        upsample_graph: E_2 x 2 [center_node, neighbor_node]
    """
    batch_x = torch.tensor([0]*len(points)).to(points.device)
    batch_y = torch.tensor([0]*len(key_points)).to(key_points.device)

    downsample_graph = radius(points, key_points, radiu, batch_x, batch_y)
    upsample_graph = radius(key_points, points, radiu, batch_y, batch_x)

    return downsample_graph, upsample_graph

def intra_level_graph(key_points: torch.Tensor, radiu, loop: bool=False):
    """
    args:
        key_points: nodes of specific level
        loop: True if node has edge to itself.

    return: self_level_graph E x 2, [center_node, neighbor_node]
    """
    batch_x = torch.tensor([0]*len(key_points)).to(key_points.device)
    intra_graph = radius_graph(key_points, radiu, batch_x, loop)
    return intra_graph


if __name__ == "__main__":
    points = torch.rand(10000, 3)*5
    voxel_size = [0.5, 0.5, 0.1]
    key_points = voxelize(points, voxel_size).float()
    print("key_points.sshape", key_points.shape)

    downsample_graph, upsample_graph = inter_level_graph(points, key_points, 1)
    print("downsample_graph.shape", downsample_graph.shape)
    print("upsample_graph.shape", upsample_graph.shape)

    intra_graph = intra_level_graph(key_points, 2)
    ### test distance
    center_nodes = key_points[self_graph[0]]
    neighbor_nodes = key_points[self_graph[1]]
    distance = ((center_nodes-neighbor_nodes)**2).sum(1)
    print("distance.shape", distance.shape)
    print(torch.sqrt(distance).max())

    print("self_graph.shape", self_graph.shape)


