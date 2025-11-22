"""
学习率优化方法

较之于 AddGrad, RMSProp,Adam, 可通过 等间隔，指定间隔，指数等方式，手动控制学习率调整

分类：
    等间隔；指定间隔；指数

等间隔：
    step_size: 间隔轮数
    gamma: 学习率衰减系数， lr= lr * gamma

指定间隔：
    milestones：设定调整轮次:[50, 125, 160]
    gamma：调整系数
指数：
    gamma：指数的底

实际中常用 Adam直接公式调整
"""

import torch
from torch import optim
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 等间隔
def step_lr():
    # 初始化参数: 初始学习率， 轮数， 迭代次数
    lr, epochs, iteration = 0.1, 200, 10
    # 真实值
    y_true = torch.tensor([0],dtype=torch.float)
    # 预测值
    x = torch.tensor([1.0],dtype=torch.float)
    # 模型参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)

    # 创建优化器
    # params: 待优化的参数； lr: 学习率； momentum: 动量
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)

    # 创建学习率调整器
    # step_size: 间隔轮数 gamma: 学习率衰减系数
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 创建两个列表，分别保存训练轮数和每轮训练用的学习率
    epoch_list, lr_list = [], []

    for epoch in range(epochs): # epoch: 0 - 199
        # 获取当前轮数和学习率， 并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        for batch in range(iteration):
            y_pred = w * x
            # 计算损失值, 最小二乘法
            loss = (y_pred - y_true) ** 2
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()


    # 绘制学习率曲线
    plt.plot(epoch_list, lr_list)
    plt.title("学习率曲线")
    plt.xlabel("轮数")
    plt.ylabel("学习率")
    plt.show()

# 2. 指定间隔
def step_lr_interval():
    # 初始化参数: 初始学习率， 轮数， 迭代次数
    lr, epochs, iteration = 0.1, 200, 10
    # 真实值
    y_true = torch.tensor([0],dtype=torch.float)
    # 预测值
    x = torch.tensor([1.0],dtype=torch.float)
    # 模型参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)

    # 创建优化器
    # params: 待优化的参数； lr: 学习率； momentum: 动量
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)

    # 创建学习率调整器
    # milestones:指定步长 gamma: 学习率衰减系数
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 125, 160], gamma=0.5)

    # 创建两个列表，分别保存训练轮数和每轮训练用的学习率
    epoch_list, lr_list = [], []

    for epoch in range(epochs): # epoch: 0 - 199
        # 获取当前轮数和学习率， 并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        for batch in range(iteration):
            y_pred = w * x
            # 计算损失值, 最小二乘法
            loss = (y_pred - y_true) ** 2
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()

    # 绘制学习率曲线
    plt.plot(epoch_list, lr_list)
    plt.title("学习率曲线")
    plt.xlabel("轮数")
    plt.ylabel("学习率")
    plt.show()

# 3. 指数
def step_lr_exponential():
    # 初始化参数: 初始学习率， 轮数， 迭代次数
    lr, epochs, iteration = 0.1, 200, 10
    # 真实值
    y_true = torch.tensor([0],dtype=torch.float)
    # 预测值
    x = torch.tensor([1.0],dtype=torch.float)
    # 模型参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)

    # 创建优化器
    # params: 待优化的参数； lr: 学习率； momentum: 动量
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)

    # 创建学习率调整器
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 创建两个列表，分别保存训练轮数和每轮训练用的学习率
    epoch_list, lr_list = [], []

    for epoch in range(epochs): # epoch: 0 - 199
        # 获取当前轮数和学习率， 并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        for batch in range(iteration):
            y_pred = w * x
            # 计算损失值, 最小二乘法
            loss = (y_pred - y_true) ** 2
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()

    # 绘制学习率曲线
    plt.plot(epoch_list, lr_list)
    plt.title("学习率曲线")
    plt.xlabel("轮数")
    plt.ylabel("学习率")
    plt.show()


if __name__ == '__main__':
    step_lr()
    step_lr_interval()
    step_lr_exponential()