import torch
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

weights = torch.tensor([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
])

bias = torch.tensor([1.0, 2.0])

res1 = x@weights + bias
print(res1)
print(res1.shape)

for index,y in enumerate(x):
    print(f"第{index+1}行平均值为{y.mean()}")
x = x.reshape(3,2)
print(torch.cuda.is_available())

# 1. 计算 x @ weights + bias，并输出结果和 shape。
# 2. 计算 x 每一行的平均值。
# 3. 将 x 重塑为形状 (3, 2)。
# 4. 输出 torch.cuda.is_available() 的结果。


# # 创建一维 Tensor
# x = torch.tensor([1.0, 2.0, 3.0])
# print(x)
# print(x.shape)
# print(x.dtype)
#
# # 创建二维 Tensor
# matrix = torch.tensor([
#     [1.0, 2.0],
#     [3.0, 4.0],
# ])
#
# print(matrix)
# print(matrix.shape)
#
# # 与 NumPy 类似的批量运算
# print(matrix + 1)
# print(matrix * 2)
#
# # 矩阵乘法
# a = torch.tensor([
#     [1.0, 2.0, 3.0],
#     [4.0, 5.0, 6.0],
# ])
#
# b = torch.tensor([
#     [0.1, 0.2],
#     [0.3, 0.4],
#     [0.5, 0.6],
# ])
#
# result = a @ b
# print(result)
# print(result.shape)  # torch.Size([2, 2])
#
# print(torch.cuda.is_available())