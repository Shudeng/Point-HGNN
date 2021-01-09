import torch
from hgnn_backbone import HGNN


def sample_indices(edges):
    """
    During propagation, for every center point,
    choose the first end point (from upsample graph) of its edges in graph as its indice.
    :param
        edges: a tensor, 2 x E, [centers, neighbors].
    :return:
        a 1-d tensor, length as the number of center points, the indices of center points.
    """
    centers = edges[0, :]
    neighbors = edges[1, :]
    unique_centers = torch.unique(centers)
    # print('number of unique centers: ', unique_centers.size(0).item())

    idxs = []
    for c in unique_centers:
        # index of the first repeated element
        idx = torch.nonzero(torch.eq(centers, c), as_tuple=False)[0, 0].item()
        idxs.append(idx)
    # print(idxs)
    return neighbors[idxs]


def build_hgnn_backbone(cfg=None):
    downsample_voxel_sizes = [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5]]
    inter_radius = [0.3, 0.5, 0.7]
    intra_radius = [0.4, 0.6, 0.8]

    max_num_neighbors = 256  # 32 is default, 256 is adopted in Point-GNN
    # (256 for training and all edges for inference)

    backbone = HGNN(downsample_voxel_sizes, inter_radius, intra_radius, max_num_neighbors)
    return backbone

if __name__ == "__main__":
    backbone = build_hgnn_backbone()
    print(backbone)

