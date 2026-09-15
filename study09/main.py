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

# 1. 创建 dataset
dataset = TensorDataset(x, target)

# 2. 创建 loader：每个 batch 取 2 条，先不要打乱
loader = DataLoader(dataset, batch_size=2,shuffle=False)

# 3. 查看总数据量和第一条数据
print("数据量：",len(dataset))
print("第一条：",dataset[0])

# 4. 遍历并打印每个 batch 的内容和形状
for batch_x, batch_y in loader:
    print("batch_x:", batch_x)
    print("batch_y:", batch_y)
    print("x 的 shape:", batch_x.shape)
    print("y 的 shape:", batch_y.shape)