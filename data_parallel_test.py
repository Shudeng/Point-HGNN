from mmcv import Config

#cfg = Config.fromfile("config/dataset/kitti-3d-3class.py")
#print(cfg)
#print(cfg.data)

import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(10,100)
        self.linear1 = nn.Linear(100, 6)

    def forward(self, x):

        return x.sum()

model = Model()
model = model.cuda()
print(0, next(model.parameters()).device)

device_ids = [0,1,4, 6]
replicas = nn.parallel.replicate(model, device_ids)

input = torch.rand(8, 2, 2)
device_ids = [0,1,2,3]
inputs = nn.parallel.scatter(input, device_ids)
print("1", inputs)
for i, input_ in enumerate(inputs):
    print(i, input_.device)


inputs = [torch.rand(2, 2, 2).cuda(0), torch.rand(2, 2, 2).cuda(1), torch.rand(2, 2, 2).cuda(2), torch.rand(2, 2, 2).cuda(3)]
print("2", inputs)
for i, input_ in enumerate(inputs):
    print(i, input_.device)

for i, input_ in enumerate(inputs):
    print(i, input.device)

output = nn.parallel.parallel_apply(replicas, inputs)
for i, out in enumerate(output):
    print(i, out.device)


"""
print(0, next(replicas[0].parameters()).device)
print(1, next(replicas[1].parameters()).device)
print(2, next(replicas[2].parameters()).device)
print(replicas)
"""


