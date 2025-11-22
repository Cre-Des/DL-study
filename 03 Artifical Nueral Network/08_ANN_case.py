"""
ANN 案例

手机价格分类案例

基于20列特征 预测4个价格区间

- 准备训练集数据
- 构建要使用的模型
- 模型训练
- 模型预测评估

优化思路：
    1. 优化方法：SGD->Adam
    2. 学习率 0.001 -> 0.0001
    3. 对数据进行标准化
    4. 增加网络深度， 调整神经元数量
    5. 调整训练轮次
"""
import torch                                            # Pytorch 框架，张量操作
import torch.nn as nn                                   # 神经网络操作
import pandas as pd                                     # 数据处理
from sklearn.model_selection import train_test_split    # 训练集和测试集划分
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset              # 数据集对象 数据-Tensor-数据集-数据加载器
from torch.utils.data import DataLoader                 # 数据加载器
import torch.optim as optim                             # 优化器
import numpy as np                                      # 数组（矩阵）操作
import time                                             # 时间模块
from torchsummary import summary                        # 模型结构可视化

# 1. 准备训练集数据
def create_dataset():
    # 1.1 读取数据
    data = pd.read_csv('./data/mobile_price.csv')
    # print(f'data: {data.head()}')
    # print(f'data: {data.shape}')

    # 1.2 获取 x 特征列和 y 标签列
    x,y = data.iloc[:, :-1], data.iloc[:, -1]

    # 1.3 类型转换
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 1.4 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88, stratify=y)

    # 优化 ：数据标准化
    transfer = StandardScaler ()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 1.5 创建数据集对象
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.tensor(y_test.values))

    # 1.6 返回 数据集对象，特征列数，标签类别数
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))

# 2. 构建模型
class MobilePriceClassifier(nn.Module):
    # 2.1 在 init 函数里面初始化父类成员， 搭建神经网络
    def __init__(self, input_dim, output_dim):
        super(MobilePriceClassifier,self).__init__()
        # 搭建神经网络
        # 优化 ： 增加网络深度
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 128)
        self.output = nn.Linear(128, output_dim)

    # 2.2 在 forward 函数里面完成前向传播
    def forward(self, x):
        # 隐藏层1 加权求和 + 激活函数(relu)
        x = torch.relu(self.linear1(x))

        # 隐藏层2 加权求和 + 激活函数(relu)
        x = torch.relu(self.linear2(x))

        # 隐藏层3 加权求和 + 激活函数(relu)
        x = torch.relu(self.linear3(x))

        # 隐藏层4 加权求和 + 激活函数(relu)
        x = torch.relu(self.linear4(x))

        # 输出层 加权求和 + 激活函数(softmax)
        # 使用多分类交叉熵损失函数 ClassEntropyLoss() 替代softmax
        x = self.output(x)
        return x

# 3. 模型训练
def train(train_dataset, input_dim, output_dim):
    # 3.1 创建数据加载器
    # 参数1：数据集对象（1600条） 参数2：批次大小 参数3：是否打乱数据
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 3.2 创建模型对象
    model = MobilePriceClassifier(input_dim, output_dim)

    # 3.3 创建损失函数对象
    criterion = nn.CrossEntropyLoss()

    # 3.4 创建优化器对象
    # optimizer = optim.SGD(params=model.parameters(), lr=0.001)
    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

    # 3.5 训练模型
    # 定义训练轮数
    epochs = 100
    for epoch in range(epochs):
        # 定义变量 当前批次数 每次训练的损失值
        total_loss, batch_num = 0.0, 0
        # 训练开始的时间
        start = time.time()
        # 本轮训练开始
        for train_x, train_y in train_loader:
            # 切换模型训练模式
            model.train() # 训练模式 model.eval() 测试模式
            # 前向传播
            y_pred = model(train_x)
            # 计算损失值
            loss = criterion(y_pred, train_y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累加损失值
            total_loss += loss.item()
            batch_num += 1

        print(f'第 {epoch+1} 轮训练结束，总损失值: {total_loss/batch_num:.4f}， 训练时间: {time.time()-start:.2f}s')

    # 保存模型（参数）
    # 参数：1. 模型对象；2. 模型参数保存路径
    # print(f'\n\n 模型的参数信息：{model.state_dict()}\n\n')
    torch.save(model.state_dict(), './model/mobile_price_classifier.pth') # .pth .pkl .pickle 均可
    return model

# 4. 模型测试
def evaluate(test_dataset, input_dim, output_dim):
    # 4.1 创建数据加载器
    # 创建数据加载器
    # 参数1：数据集对象（400条） 批次大小 是否打乱数据
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 4.2 创建模型对象
    model = MobilePriceClassifier(input_dim, output_dim)

    # 4.3 加载模型参数
    model.load_state_dict(torch.load('./model/mobile_price_classifier.pth'))

    # 4.4 模型测试
    # 定义变量 正确预测数
    correct = 0

    # 4.5 测试开始
    for test_x, test_y in test_loader:
        # 切换模型测试模式
        model.eval()
        # 前向传播
        y_pred = model(test_x)
        # 输出层没有 softmax 根据加权求和，得到类别，用argmax() 获取最大值的下标就是类别
        y_pred = torch.argmax(y_pred, dim=1) # dim = 1 逐行处理
        print(f'预测类别: {y_pred}')

        # 累加正确预测数
        correct += (y_pred == test_y).sum()

    print(f'测试集 准确率(Accuracy): {correct/len(test_dataset):.4f}')


# 5. 测试
if __name__ == '__main__':
    print('================准备数据=================')
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f'训练集 数据集对象: {train_dataset}')
    print(f'测试集 数据集对象: {test_dataset}')
    print(f'输入特征数: {input_dim}')
    print(f'标签类别数: {output_dim}')
    print('================构建模型=================')
    model = MobilePriceClassifier(input_dim, output_dim)
    # 计算模型参数
    # 参数：1. 模型对象；2. 输入数据维度(批次大小，输入特征数)
    summary(model, input_size=(16,input_dim))
    print('================模型训练=================')
    model = train(train_dataset, input_dim, output_dim)
    print('================模型测试=================')
    evaluate(test_dataset, input_dim, output_dim)
