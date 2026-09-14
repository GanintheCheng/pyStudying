import numpy as np

x = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

weights = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
])

bias = np.array([1.0, 2.0])

res = x@weights + bias
print(res)
print(res.shape)

X = x.reshape(6)
print(X.shape)

XX = np.reshape(X,(3,2))

#因为矩阵运算原理前面的矩阵的列要和后面矩阵的行相同

# 请完成：
# 1. 计算 x @ weights + bias 并打印结果与形状。
# 2. 将 x 重塑成一维数组，打印形状。
# 3. 将重塑后的一维数组再变为 (3, 2)。
# 4. 用一句话解释为什么 (2, 3) @ (3, 2) 能计算，但 (2, 3) @ (2, 3) 不能计算。