"""
基于自动微分实现的线性回归案例

使用 nn.MSELoss() 代替平方损失函数
使用 data.DataLoader 代替数据加载器
使用 optim.SGD 代替优化器
使用 nn.Linear 代替假设函数

"""

# import necessary lib
import torch
from scipy.ndimage import label
from torch.utils.data import TensorDataset # 构造数据集对象
from torch.utils.data import DataLoader # 数据加载器
from torch import nn # nn 模块中有平方损失函数和假设函数
from torch import optim # optim 模块中有优化函数
from sklearn.datasets import make_regression # 创建线性回归模型数据集
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 定义函数, 创建线性回归样本数据
def create_dataset():
    # 创建数据集
    x, y, coef = make_regression(
        n_samples=100, # 100条样本(100个样本点）
        n_features=1, # 1个特征（1个特征点）
        noise= 10, # 噪声，噪声越大，样本点越散，噪声越小，样本点越集中
        coef=True, # 是否返回系数，默认为False，返回值为None
        bias=14.5, # 偏置
        random_state=23 # 随机种子，随机种子相同，输出数据相同
    )

    # 封装为张量
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    return x, y, coef

# 2. 模型训练
def train(x,y,coef):
    # 创建数据集对象
    dataset = TensorDataset(x, y)

    # 创建数据加载器对象
    # 参数1: 数据集对象; 参数2: 批次大小; 参数3: 是否打乱数据(训练集打乱, 测试集不打乱)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 创建初始的线性回归模型
    model = nn.Linear(1, 1)

    # 创建损失函数对象
    criterion = nn.MSELoss()

    # 创建优化器对象
    # 1. 模型参数; 2. 学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 具体训练过程
    # 1. 定义: 训练轮数, 每轮(平均)损失值, 训练总损失值, 训练样本数.
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0
    # 2. 开始训练
    for epoch in range(epochs):
        for train_x, train_y in dataloader:
            # 模型预测
            y_pred = model(train_x)
            # 计算损失
            loss = criterion(y_pred, train_y.reshape(-1, 1)) # -1 自动计算
            # 计算总损失
            total_loss += loss.item()
            total_sample += 1
            # 梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 反向传播, 计算梯度
            optimizer.step()        # 梯度更新

        # 把本轮的平均损失值添加进列表中
        loss_list.append(total_loss / total_sample)
        print(f'轮数: {epoch}, 平均损失: {total_loss / total_sample}')

    print(f'{epochs}轮的平均损失分别为: {loss_list}')
    print(f'模型参数:{model.weight},\n 偏置:{model.bias}')

    # 绘制损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title("损失函数曲线变化图")
    plt.grid(True)
    plt.show()

    # 绘制样本点分布情况
    plt.scatter(x,y)
    # 预测值
    y_pred = torch.tensor(data= [v * model.weight.detach() + model.bias.detach() for v in x])
    # 真实值
    y_true = torch.tensor(data= [v * torch.tensor(coef) + 14.5 for v in x])

    plt.plot(x, y_pred, color = 'red', label = '预测值')
    plt.plot(x, y_true, color = 'green', label = '真实值')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    x, y, coef = create_dataset()
    print(f'x = {x}, \ny = {y}, \ncoef = {coef}')
    train(x, y, coef)




