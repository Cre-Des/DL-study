"""
池化层相关操作

目的：降维

最大池化 MaxPool2d(kernel_size=2, stride=1, padding=0)
平均池化 AvgPool2d(kernel_size=2, stride=1, padding=0)

不改变数据通道数
"""

import torch
import torch.nn as nn

# 1. 定义函数，演示单通道池化
def single_channel_pooling():
    # 创建 1通道， 3*3的矩阵
    input = torch.tensor([[
        [0,1,2],
        [3,4,5],
        [6,7,8]
    ]])
    print(f'input:\n{input}, \nshape: {input.shape}')

    # 创建最大池化层
    # 参数：
    #     kernel_size: 池化核大小
    #     stride: 步长
    #     padding: 填充
    max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output = max_pool(input)
    print(f'output:\n{output}, \nshape: {output.shape}')

    # 创建平均池化层
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    output = avg_pool(input)
    print(f'output:\n{output}, \nshape: {output.shape}')

# 2. 定义函数，演示多通道池化
def multi_channel_pooling():
    # 创建 3通道， 3*3的矩阵
    inputs = torch.tensor([[ # 3通道 C
        [0,1,2],            # 通道1 HW 3，3
        [3,4,5],
        [6,7,8]
    ],

    [                       # 通道2 HW 3，3
        [10,20,30],
        [40,50,60],
        [70,80,90]
    ],

    [                       # 通道3 HW 3，3
        [11,22,33],
        [44,55,66],
        [77,88,99]
    ]])

    # 创建最大池化层
    # 参数：
    #     kernel_size: 池化核大小
    #     stride: 步长
    #     padding: 填充
    max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output = max_pool(inputs)
    print(f'output:\n{output}, \nshape: {output.shape}')

    # 创建平均池化层
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    output = avg_pool(inputs)
    print(f'output:\n{output}, \nshape: {output.shape}')

# 3. 测试
if __name__ == "__main__":
    # single_channel_pooling()
    multi_channel_pooling()