"""
RNN 层的使用
"""

import torch
import torch.nn as nn

# 创建循环网络层
# 输入参数： 1. 词向量的维度x_t（RNN看到的细节）； 2. 隐藏状态向量维度h（类比大脑能记住的细节）； 3. 隐藏层数数量
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)

# 创建输入数据,表示输入的x
# 输入数据维度： 1. 每个句子词的个数（共5帧画面）； 2. 句子的数量B（32个人同时看这五帧）； 3. 词向量的维度x_t （看到的细节）
inputs = torch.randn(5, 32, 128)

# 创建隐藏状态 h
# 隐藏状态维度： 1. 隐藏层数量 = num_layers； 2. 句子的数量B； 3. 隐藏状态向量维度h
hidden = torch.zeros(1, 32, 256)

# 调用RNN处理

# 返回值1： 输出结果 所有时间步的输出，包含所有时间步的隐藏状态
# 返回值2： 隐藏状态 最后一层隐藏状态
output, hidden = rnn(inputs, hidden)
print(f'output.shape: {output.shape}')
print(f'hidden.shape: {hidden.shape}')
