"""
参数初始化

无法打破对称性
- 全0初始化(记忆)
    nn.init.zeros_()
- 全1初始化
    nn.init.ones_()
- 固定值初始化
    nn.init.constant_()
可以打破对称性
- 随机初始化
    nn.init.uniform_()
    nn.init.normal_()

- kaiming 初始化(+ ReLU) 专为ReLU和其变体设计，考虑到ReLU激活函数的特性，对输入维度进行缩放(记忆)
    正态分布的 he初始化
        nn.init.kaiming_normal_()
    均匀分布的 he初始化
        nn.init.kaiming_uniform_()
- xavier初始化  适用于Sigmoid、Tanh 等激活函数，解决梯度消失问题(记忆)
    正态分布的 xavier初始化
        nn.init.xavier_normal_()
    均匀分布的 xavier初始化
        nn.init.xavier_uniform_()

激活函数 ReLU 及其系列，优先 kaiming 初始化
激活函数非 ReLU，优先 xavier初始化

若为浅层网络，可以考虑随机初始化
"""

import torch.nn as nn
from PIL.GimpGradientFile import linear


# 均匀分布随机初始化
def uniform_init():
    # 创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 初始化w
    nn.init.uniform_(linear.weight)
    # 初始化b
    nn.init.uniform_(linear.bias)

    print(linear.weight.data) # .data
    print(linear.bias.data)

# 正态分布随机初始化 默认均值为0，方差为1
def normal_init():
    # 创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 初始化w
    nn.init.normal_(linear.weight)
    # 初始化b
    nn.init.normal_(linear.bias)

    print(linear.weight.data) # .data
    print(linear.bias.data)

# 全0初始化
def zero_init():
    # 创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 初始化w
    nn.init.zeros_(linear.weight)
    # 初始化b
    nn.init.zeros_(linear.bias)

    print(linear.weight.data) # .data
    print(linear.bias.data)

# 全1初始化
def ones_init():
    # 创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 初始化w
    nn.init.ones_(linear.weight)
    # 初始化b
    nn.init.ones_(linear.bias)

    print(linear.weight.data) # .data
    print(linear.bias.data)

# 固定值初始化
def constant_init():
    # 创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 初始化w
    nn.init.constant_(linear.weight,3)
    # 初始化b
    nn.init.constant_(linear.bias,3)

    print(linear.weight.data) # .data
    print(linear.bias.data)

# kaiming 初始化
# 正态分布的he初始化
def kaiming_normal_init():
    # 创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 初始化w
    nn.init.kaiming_normal_(linear.weight)
    print(linear.weight.data) # .data
# 均匀分布的he初始化
def kaiming_uniform_init():
    # 创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 初始化w
    nn.init.kaiming_uniform_(linear.weight)
    print(linear.weight.data) # .data

# xavier初始化
# 正态分布的 xavier初始化
def xavier_normal_init():
    # 创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 初始化w
    nn.init.xavier_normal_(linear.weight)
    print(linear.weight.data) # .data

# 均匀分布的 xavier初始化
def xavier_uniform_init():
    linear = nn.Linear(5, 3)
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight.data)


if __name__ == '__main__':
    # uniform_init()
    # constant_init()
    # zero_init()
    # ones_init()
    # normal_init()
    # kaiming_normal_init()
    # kaiming_uniform_init()
    # xavier_normal_init()
    xavier_normal_init()

