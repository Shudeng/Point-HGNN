from torch_cluster import radius, radius_graph
import torch

x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
batch_x = torch.tensor([0, 0, 0, 0])
y = torch.Tensor([[-1, 0], [1, 0]])
batch_y = torch.tensor([0, 0])
assign_index = radius(x, y, 1.5, batch_x, batch_y)
print(assign_index)

assign_index = radius(y, x, 1.5, batch_y, batch_x)
print(assign_index)

edge_index = radius_graph(x, r=2, batch=batch_x, loop=False)
print(edge_index)

