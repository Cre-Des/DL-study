"""
损失函数演示

分类问题：
    多分类交叉熵损失函数  CrossEntropyLoss
    二分类交叉熵损失函数  BCELoss
回归问题：
    MAE (Mean Absolute Error) L1: 平均绝对误差
    MSE (Mean Squared Error)  L2: 均方误差
    Smooth L1 : 结合上两个特点的升级优化 (优先)
        - 在[-1,1]之间实际上就是L2损失，这样解决了L1的不光滑问题
        - 在[-1,1]区间外，实际上就是L1损失，这样就解决了离群点梯度爆炸的问题
"""

import torch
import torch.nn as nn

# 1. 定义多分类交叉熵损失函数
def cross_entropy_loss():
    # y_true = torch.tensor([[0,1,0],[1,0,0]],dtype=torch.float)
    y_true = torch.tensor([1,2])

    y_pred = torch.tensor([[0.1,0.8,0.1],[0.7,0.2,0.1]],requires_grad=True,dtype=torch.float)

    criterion = nn.CrossEntropyLoss() #reduction='mean'
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')

# 2. 定义二分类交叉熵损失函数
def bce_loss():
    y_true = torch.tensor([0,1,0],dtype=torch.float)
    y_pred = torch.tensor([0.6901, 0.5459, 0.2469], requires_grad=True)

    criterion = nn.BCELoss() # reduction='mean'
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')

# 3. 定义MAE
def mae_loss():
    y_true = torch.tensor([2.0,2.0,2.0],dtype=torch.float)
    y_pred = torch.tensor([1.0,1.0,1.9],requires_grad=True,dtype=torch.float)

    criterion = nn.L1Loss()
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')

# 4. 定义MSE
def mse_loss():
    y_true = torch.tensor([2.0,2.0,2.0],dtype=torch.float)
    y_pred = torch.tensor([1.0,1.0,1.9],requires_grad=True,dtype=torch.float)

    criterion = nn.MSELoss()
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')

# 5. 定义SmoothL1
def SmoothL1_loss():
    y_true = torch.tensor([2.0,2.0,2.0],dtype=torch.float)
    y_pred = torch.tensor([1.0,1.0,1.9],requires_grad=True,dtype=torch.float)

    criterion = nn.SmoothL1Loss()
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')

if __name__ == '__main__':
    # cross_entropy_loss()
    # bce_loss()
    # mae_loss()
    # mse_loss()
    SmoothL1_loss()