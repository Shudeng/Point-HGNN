import torch


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
    idxs = []
    for c in torch.unique(centers):
        # index of the first repeated element
        idx = torch.nonzero(torch.eq(centers, c), as_tuple=False)[0, 0].item()
        idxs.append(idx)
    # print(idxs)
    return neighbors[idxs]