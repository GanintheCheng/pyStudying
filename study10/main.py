import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


torch.manual_seed(42)

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
])

target = torch.tensor([
    [1.0],
    [4.0],
    [7.0],
    [10.0],
])

criterion = nn.MSELoss()
model = LinearModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

dataset = TensorDataset(x, target)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model.train()

for epoch in range(100):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        prediction = model(batch_x)
        loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()
print(f'weight={model.linear.weight}, bias={model.linear.bias}')
model.eval()
with torch.no_grad():
    prediction = model(torch.tensor([
        [5.0],
        [10.0],
    ]))
print(prediction)
