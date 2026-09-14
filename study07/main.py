import torch

import torch

weight = torch.tensor(0.0, requires_grad=True)

x = torch.tensor(3.0)
target = torch.tensor(10.0)
learning_rate = 0.01

for epoch in range(20):
    # 1. 前向计算 prediction
    # 2. 计算 loss
    # 3. loss.backward()
    # 4. 在 torch.no_grad() 中更新 weight
    # 5. 清空 weight.grad
    # 6. 打印 epoch、loss、weight
    prediction = weight * x
    loss = (prediction - target)**2
    loss.backward()
    with torch.no_grad():
        weight -= learning_rate * weight.grad
    weight.grad.zero_()
    print(epoch, loss,weight)

# # 需要学习的权重
# weight = torch.tensor(2.0, requires_grad=True)
#
# # 一条输入数据，以及正确答案
# x = torch.tensor(3.0)
# target = torch.tensor(10.0)
#
# # 模型预测：prediction = x * weight
# prediction = x * weight
#
# # 损失：预测值与真实值的平方差
# loss = (prediction - target) ** 2
#
# # 自动计算 loss 对 weight 的梯度
# loss.backward()
#
# print(f"预测值: {prediction.item()}")
# print(f"损失: {loss.item()}")
# print(f"weight 的梯度: {weight.grad.item()}")
#
# while loss > 0.00000001:
#     prediction = x * weight
#     loss = (prediction - target) ** 2
#     loss.backward()
#     with torch.no_grad():
#         weight -= 0.01 * weight.grad
#     weight.grad.zero_()
#     print(weight)
#
# print(weight)
