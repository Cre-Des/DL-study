"""
图片分类案例

- 准备训练集数据
- 构建要使用的模型
- 模型训练
- 模型预测评估

模型结构：
1. 输入形状: 32x32
2. 第一个卷积层输入 3 个 Channel, 输出 6 个 Channel, Kernel Size 为: 3x3
3. 第一个池化层输入 30x30, 输出 15x15, Kernel Size 为: 2x2, Stride 为: 2
4. 第二个卷积层输入 6 个 Channel, 输出 16 个 Channel, Kernel Size 为 3x3
5. 第二个池化层输入 13x13, 输出 6x6, Kernel Size 为: 2x2, Stride 为: 2
6. 第一个全连接层输入 576 (16*6*6)维, 输出 120 维
7. 第二个全连接层输入 120 维, 输出 84 维
8. 最后的输出层输入 84 维, 输出 10 维

卷积核参数数量计算公式：
输入通道数 * 卷积核尺寸（F*F）*卷积核数量 + 卷积核数量
"""
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8

# 1. 准备训练集数据
def create_dataset():
    # 1.1 创建训练集对象
    # 参数1：数据集存放路径 参数2：是否为训练集 参数3：标签转换成张量 参数4：是否联网下载（若无数据集则下载）
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    # 1.2 创建测试集对象
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    # 1.3 返回 训练集对象，测试集对象
    return train_dataset, test_dataset

# 2. 构建模型
class CIFAR10Classifier(nn.Module):
    # 2.1 在 init 函数里面初始化父类成员， 搭建神经网络
    def __init__(self):
        super(CIFAR10Classifier,self).__init__()
        # 搭建神经网络
        self.conv1 = nn.Conv2d(3, 6, 3, padding=0, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=0, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        # 第一层 ： 卷积层(加权求和) + 激活函数(relu) + 池化层(降维)
        x = self.pool1(torch.relu(self.conv1(x)))
        # 第二层 ： 卷积层(加权求和) + 激活函数(relu) + 池化层(降维)
        x = self.pool2(torch.relu(self.conv2(x)))

        # 全连接层： 输入 576 (16*6*6)维，输出 120 维
        # 只能处理二维数据，所以需要将数据拉平，才能进行全连接 (8,16,6,6) BATCH_SIZE = 8 --> (8,576)
        x = x.reshape(x.size(0),-1)
        # print(f'全连接层输入: {x.shape}')

        x = torch.relu(self.fc1(x))
        # 全连接层： 输入 120 维，输出 84 维
        x = torch.relu(self.fc2(x))
        # 输出层： 输入 84 维，输出 10 维
        x = self.output(x)

        return x

# 3. 模型训练
def train(train_dataset):
    # 3.1 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 3.2 创建模型对象
    model = CIFAR10Classifier()
    # 3.3 创建损失函数对象
    criterion = nn.CrossEntropyLoss()
    # 3.4 创建优化器对象
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    # 3.5 训练模型
    epochs = 10
    for epoch in range(epochs):
        # 定义变量 总训练损失值，总样本数据量，预测正确样本个数，训练（开始时间）
        total_loss , total_samples, correct_num, start = 0.0, 0, 0, time.time()
        for train_x, train_y in train_loader:
            # 训练模式
            model.train()
            # 前向传播
            y_pred = model(train_x)
            # 计算损失值
            loss = criterion(y_pred, train_y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计预测正确的样本个数
            # print(y_pred.argmax(dim=1))
            correct_num += (torch.argmax(y_pred, dim=-1) == train_y).sum().item()

            # 当前批次的损失值
            total_loss += loss.item() * len(train_y)
            total_samples += len(train_y)

        # 打印当前轮训练结果
        print(f'第 {epoch+1} 轮训练完成，总损失值: {total_loss:.5f}，总样本数: {total_samples}，准确率: {correct_num/total_samples:.2%}，训练时间: {time.time()-start:.2f}s')

    # 保存模型
    torch.save(model.state_dict(), './model/cifar10_classifier.pth')

# 4. 模型测试
def evaluate(test_dataset):
    # 4.1 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 4.2 创建模型对象
    model = CIFAR10Classifier()
    model.load_state_dict(torch.load('./model/cifar10_classifier.pth'))
    # 4.3 测试模型
    total_samples, correct_num = 0, 0
    for test_x, test_y in test_loader:
        # 测试模式
        model.eval()
        # 前向传播
        y_pred = model(test_x)
        # 使用了CrossEntropyLoss 训练时没有用 softmax 激活函数，所以这里使用 argmax
        y_pred = torch.argmax(y_pred, dim=-1)
        # 统计预测正确的样本个数
        correct_num += (torch.argmax(y_pred, dim=-1) == test_y).sum().item()
        total_samples += len(test_y)

    # 打印测试结果
    print(f'测试结果：ACC: {correct_num/total_samples:.2}')

# 5. 测试
if __name__ == '__main__':
    print('================准备数据=================')
    train_dataset, test_dataset = create_dataset()
    print(f'训练集: {train_dataset.data.shape}')
    print(f'测试集: {test_dataset.data.shape}')
    print(f'数据集类别：{train_dataset.class_to_idx}')

    # 图像展示
    # plt.figure(figsize=(2,2))
    # plt.imshow(train_dataset.data[11])
    # plt.title(train_dataset.targets[11])
    # plt.show()

    print('================构建模型=================')
    model = CIFAR10Classifier()
    # 计算模型参数
    # 参数：1. 模型对象；2. 输入数据维度(CHW) 3. 批次大小
    summary(model, input_size=(3,32,32),batch_size=BATCH_SIZE, device='cpu')
    print('================模型训练=================')
    # train(train_dataset)
    print('================模型测试=================')
    evaluate(test_dataset)

