import torch
from mmdet3d.ops import Points_Sampler

# sampler = Points_Sampler(num_point=[5, 3], fps_mod_list=['D-FPS']*2, fps_sample_range_list=[-1, -1]) 
sampler = Points_Sampler(num_point=[3], fps_mod_list=['D-FPS'], fps_sample_range_list=[-1]) 
points_xyz = torch.rand(1, 10, 3).cuda()
print(points_xyz)
features = torch.rand(1, 5, 10).cuda()
print(features)

indices = sampler(points_xyz, features)
print(indices.size())
print(indices.dtype)
print(indices.device)
print(indices.cpu())

