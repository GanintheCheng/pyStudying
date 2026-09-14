import torch

if __name__ == '__main__':
    weight = torch.tensor(0.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([3.0, 5.0, 7.0, 9.0])

    learning_rate = 0.01

    for epoch in range(100):
        # TODO 1：计算 prediction，模型公式为 y = x * weight + bias
        prediction = x * weight + bias
        # TODO 2：计算均方误差 loss
        # 提示：((prediction - target) ** 2).mean()
        loss = (prediction - target).pow(2).mean()
        # TODO 3：反向传播
        loss.backward()
        # TODO 4：在 torch.no_grad() 中更新 weight 和 bias
        with torch.no_grad():
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad
            # TODO 5：清空 weight、bias 的旧梯度
        weight.grad.zero_()
        bias.grad.zero_()
        if epoch % 10 == 0:
            print(epoch, loss.item(), weight.item(), bias.item())
