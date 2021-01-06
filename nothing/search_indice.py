from torch_cluster import radius, radius_graph
import torch

torch.manual_seed(12345)
x = torch.rand(8, 3)
y = torch.rand(4, 3)
z = torch.rand(2, 3)
batch_x = torch.tensor([0] * 8)
batch_y = torch.tensor([0, 0, 0, 0])
batch_z = torch.tensor([0, 0])
xy = radius(x, y, 0.5, batch_x, batch_y)
yz = radius(y, z, 1, batch_y, batch_z)
print('xy:', xy)
print('yz:', yz)

def sample_indices(edges):
    centers = edges[0, :]
    neighbors = edges[1, :]
    unique_centers = torch.unique(centers)
    idxs = []
    for c in unique_centers:
        idx = torch.nonzero(torch.eq(centers, c), as_tuple=False)[0, 0].item()  # index of the first repeated element
        # print(idx)
        idxs.append(idx)
    print(idxs)
    return neighbors[idxs]

indices_x = torch.arange(8)
print('indices_x:', indices_x)
indices_y = sample_indices(xy)
print('indices_y:', indices_y)
indices_z = sample_indices(yz)
print('indices_z:', indices_z)
indices_z = indices_y[indices_z]
print('indices_z:', indices_z)

# device test
# print('==============')
# xy_cuda = xy.cuda()
# indices_y = sample_indices(xy_cuda)
# print(indices_y)
# print(indices_y.device)

# xy_cpu = xy
# indices_y = sample_indices(xy)
# print(indices_y)
# print(indices_y.device)



# # indice1, inverse_indice = torch.unique(x_idx, return_inverse=True)
# # print(indice1, inverse_indice)
# # indice2 = torch.unique(xy, dim=1)
# # print(indice2)

# # edge_index = radius_graph(x, r=2, batch=batch_x, loop=False)
# # print(edge_index)

# t = torch.tensor([[1,2],[3,4]])
# r1 = torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
# print(r1)
# r2 = torch.gather(t, 1, torch.tensor([[0],[1]]))
# print(r2)

# t = torch.tensor([1, 2, 3, 4])
# r1 = torch.gather(t, 0, torch.tensor([0, 1, 3]))
# print(r1)

# centers = torch.Tensor([1, 3, 5, 6, 8, 3])
# c = 6
# idx = torch.nonzero(torch.eq(centers, c), as_tuple=False)[0, 0].item()
# print(idx)



