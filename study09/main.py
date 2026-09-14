import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
])

target = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
])

dataset = TensorDataset(x,target)

print(len(dataset))
print(dataset[0])