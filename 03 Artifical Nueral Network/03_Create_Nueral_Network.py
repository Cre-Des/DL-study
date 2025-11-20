"""
构建如下的神经网络模型:

Input   H1   H2   Output
x_1 -  o - - o - - o -> output1
x_2 -  o - - o - - o -> output2
x_3 -  o -   +1
+1     +1
各层为全连接， o表示神经元

- 第1个隐藏层：权重初始化采用标准化的xavier初始化 激活函数使用sigmoid
- 第2个隐藏层：权重初始化采用标准化的He初始化 激活函数采用relu
- out输出层线性层 假若多分类，采用softmax做数据归一化

神经网络搭建流程：
1. 定义一个类，继承:`nn.Module`。
2. 在`__init__()`方法中，搭建神经网络。
3. 在`forward()`方法中，完成前向传播。

"""

import torch
import torch.nn as nn
from torchsummary import summary # 计算模型参数，查看模型结构

# 1. 搭建神经网络，自定义继承 nn.Module
class ModelDemo(nn.Module):
    # 1 在 init 魔法方法中，完成 初始化父类成员 及 神经网络搭建
    def __init__(self):
        # 1.1 初始化父类成员
        super().__init__()
        # 1.2 搭建神经网络，隐藏层 + 输出层
        # 隐藏层1：输入特征数3，输出特征数3，激活函数：Sigmoid
        self.linear1 = nn.Linear(3, 3)
        # 隐藏层2：输入特征数3，输出特征数2，激活函数：ReLU
        self.linear2 = nn.Linear(3, 2)
        # 输出层：输入特征数2，输出特征数2
        self.output = nn.Linear(2, 2)

        # 1.3 对隐藏层参数初始化
        # 隐藏层1
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        # 隐藏层2
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    # 2 前向传播： Input - Hide - Output
    def forward(self, x):
        # 2.1 第一层 隐藏层 加权求和 + 激活函数(Sigmoid)
        # 分解版
        # x = self.linear1(x)     # 加权求和
        # x = torch.sigmoid(x)    # 激活函数

        # 合并版
        x = torch.sigmoid(self.linear1(x))

        # 2.2 第二层 隐藏层 加权求和 + 激活函数(ReLU)
        x = torch.relu(self.linear2(x))

        # 2.3 第三层 输出层 加权求和 + 激活函数(Softmax)
        # dim = -1 按行计算，一条样本一条样本处理
        x = torch.softmax(self.output(x), dim=-1)

        return x

# 2. 模型训练
def train():
    # 1. 创建模型对象
    my_model = ModelDemo()
    # print(f'my_model: {my_model}')

    # 2. 创建数据集样本，随机生成
    data = torch.randn(5, 3)
    # print(f'data: {data}')
    # print(f'data.shape: {data.shape}')
    # print(f'data.requires_grad: {data.requires_grad}')

    # 3. 调用模型训练
    output = my_model(data) # 底层自动调用 forward(), 进行前向传播
    # print(f'output: {output}')
    # print(f'output.shape: {output.shape}')
    # print(f'output.requires_grad: {output.requires_grad}')

    # 4. 计算和查看模型参数
    print('================计算模型参数=================')
    # 参数：1. 模型对象；2. 输入数据维度（5，3）
    summary(my_model, input_size=(5, 3),batch_size=5)

    print('================查看模型参数=================')
    for name, param in my_model.named_parameters():
        print(f'name: {name}\nparam: {param}')

if __name__ == '__main__':
    train()