import torch

a = torch.rand(20, 3) * 6
print('ori', a)
voxel = torch.tensor([0.5, 1, 2])
print('voxel', voxel)

b = a / voxel
b = b.long()
print('int', b)

mod = a % voxel
print('mod', mod)

b_unique = torch.unique(b, dim=0)
print('unique int', b_unique)
print(b_unique.size())
real_b = []
for c in b_unique:
    # print('c', c)
    for i in range(len(b)):
        if torch.equal(b[i, :], c):
            real_b.append(a[i, :].unsqueeze(0))
            break
    # print(torch.equal(b, c)) 
    # idx = torch.nonzero(torch.equal(b, c), as_tuple=False)
    # print(idx)
    # [0, 0].item()
    # idxs.append(idx)
restore_b = b_unique * voxel

print(restore_b)
# print(a[idx])

print(real_b)
# print(torch.cat(real_b, dim=1))
print(len(real_b))
print(torch.cat(real_b, dim=0))

print(len(b))
print(b.size(0))

