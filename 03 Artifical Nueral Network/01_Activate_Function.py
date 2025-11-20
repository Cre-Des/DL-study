"""
各类激活函数的演示

- Sigmoid
    主要用于二分类的输出层，适用于浅层神经网络(不超过5层)
    数据在 [-6,6]有效果，在[-3, 3]效果明显

- Tanh
    主要用于隐藏层，适用于浅层神经网络(不超过5层)
    在 [-3,3]有效果， 在[-1, 1]效果明显
    求导后范围在 [0, 1], 较之 Sigmoid 收敛更快

-ReLU
    max(0,x), 计算量相对较小，训练成本低，多用于隐藏层，适合深层神经网络
    求导后要么是0，要么是1，较之Tanh收敛更快
    默认情况下 ReLU 只考虑正样本，可以使用LeakyReLU，PReLU 来考虑正负样本

- Softmax
    将多分类的结果以概率的形式展示，且概率和相加为1， 最终选取概率值最大的分类做为最终结果。

**隐藏层** 常用的激活函数(从左到右优先级递减)
ReLU, LeakyReLU, PReLU, Tanh, Sigmoid(少用)
如果你使用了ReLU 需要注意Dead ReLU问题, 避免出现0梯度而导致过多的神经元死亡

**输出层** 常用的激活函数
二分类选择 sigmoid
多分类选择 Softmax
回归问题选择 identity
"""

import torch
import matplotlib.pyplot as plt
from mpmath import scorergi
from sympy.stats.rv import probability

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

x = torch.linspace(-20, 20, 1000)
x_diff = torch.linspace(-20, 20, 1000, requires_grad=True)

# 1. Sigmoid
y = torch.sigmoid(x)
torch.sigmoid(x_diff).sum().backward()

_, axes = plt.subplots(1, 2)
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title("Sigmoid函数图像")
axes[1].plot(x, x_diff.grad)
axes[1].grid()
axes[1].set_title("Sigmoid导数图像")
plt.show()

# 2. Tanh
x_diff.grad.zero_()
y = torch.tanh(x)
torch.tanh(x_diff).sum().backward()

_, axes = plt.subplots(1, 2)
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title("Tanh函数图像")
axes[1].plot(x, x_diff.grad)
axes[1].grid()
axes[1].set_title("Tanh导数图像")
plt.show()

# 3. ReLU
x_diff.grad.zero_()
y = torch.relu(x)
torch.relu(x_diff).sum().backward()

_, axes = plt.subplots(1, 2)
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title("ReLU函数图像")
axes[1].plot(x, x_diff.grad)
axes[1].grid()
axes[1].set_title("ReLU导数图像")
plt.show()

# 4. Softmax
scores = torch.tensor([0.2,0.02,0.15,0.15,1.3, 0.5,0.06,1.1,0.05,3.75])

probabilities = torch.softmax(scores, dim=0)
print(probabilities)