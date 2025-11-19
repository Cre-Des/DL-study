'''
案例：
    演示自动微分的真实应用场景。

结论：
    1. 先前向转播（正向传播）计算出预测值(z)
    2. 基于损失函数，结合预测值(z)和真实值(y)，来计算梯度。
    3. 结合权重更新公式: W新=W旧-学习率*梯度，来更新权重。
'''
import torch

x = torch.ones(2, 5)
print(f'x = {x}')

y = torch.zeros(2, 3)
print(f'y = {y}')

w = torch.randn(5, 3, requires_grad = True)
print(f'w = {w}')

b = torch.randn(2, 3, requires_grad = True)
print(f'b = {b}')

# 前向传播
z = torch.matmul(x, w) + b
print(f'z = {z}')

# 定义损失函数
criterion = torch.nn.MSELoss()
loss = criterion(z,y)
print(f'loss: {loss}')

# 自动微分, 反向传播
loss.backward()

print(f'w 的梯度:{w.grad}')
print(f'b 的梯度:{b.grad}')

learn_rate = 0.01

w = w - learn_rate * w.grad
b = b - learn_rate * b.grad
print(f'w = {w}')
print(f'b = {b}')

