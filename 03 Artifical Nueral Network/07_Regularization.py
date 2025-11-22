"""
正则化

方式：
L1正则化：权重可以变为0
L2正则化：权重可以无限接近0
Dropout：随机失活
BN(批量归一化)
    先对数据做标准化（丢失部分信息），再对数据做缩放和平移
    BatchNorm1d：主要应用于全连接层或处理一维数据的网络，例如文本处理。它接收形状为 (N, num_features) 的张量作为输入。
    BatchNorm2d：主要应用于卷积神经网络，处理二维图像数据或特征图。它接收形状为 (N, C, H, W) 的张量作为输入。
    BatchNorm3d：主要用于三维卷积神经网络 (3D CNN)，处理三维数据，例如视频或医学图像。它接收形状为 (N, C, D, H, W) 的张量作为输入。

"""
import torch
import torch.nn as nn

# 1. Dropout
def dropout():
    # 创建隐藏层输出结果
    t1 = torch.randint(0, 10, size = (1,4)).float()

    # 进行下一层加权求和计算
    linear1 = nn.Linear(4, 5)

    # 加权求和
    l1 = linear1(t1)
    print(f'l1: {l1}')

    # 激活函数
    output = torch.relu(l1)
    print(f'output: {output}')

    # 对激活值进行随机失活处理
    dropout = nn.Dropout(p=0.4) # p: 失活概率
    d1 = dropout(output)
    print(f'随机失活后的数据: {d1}')

# 2. BN 处理二维数据
def bn():
    # 创建图像样本数据
    input_2d = torch.randn(size = (1,2,3,4))
    print(f'input_2d: {input_2d}')

    # 创建BN层
    # num_features: 输入的维度（图片的通道数）; eps: 避免除零; momentum: 动量, 用于计算移动平均统计量; affine: 是否可学习
    bn2d = nn.BatchNorm2d(num_features = 2, eps=1e-05, momentum=0.1, affine=True)

    # 进行BN处理
    output_2d = bn2d(input_2d)
    print(f'output_2d: {output_2d}')

# 3. BN 处理1维数据
def bn_1d():
    # 创建图像样本数据
    input_1d = torch.randn(size = (2,2))
    print(f'input_1d: {input_1d}')

    # 创建线性层
    linear = nn.Linear(2, 4)
    l1 = linear(input_1d)
    print(f'l1: {l1}')

    # 创建BN层
    # num_features: 输入的维度（图片的通道数）; eps: 避免除零; momentum: 动量, 用于计算移动平均统计量; affine: 是否可学习
    bn1d = nn.BatchNorm1d(num_features = 4, eps=1e-05, momentum=0.1, affine=True)

    # 进行BN处理
    output_1d = bn1d(l1)
    print(f'output_1d: {output_1d}')

if __name__ == '__main__':
    # dropout()
    # bn()
    bn_1d()