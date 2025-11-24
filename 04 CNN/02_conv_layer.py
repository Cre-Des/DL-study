"""
卷积层 用于提取局部特征，获取特征图

特征图计算：
    N = (W - F + 2P)/S + 1
        N: 输出特征图数量
        W: 输入图像宽度
        F: 卷积核大小
        P: 填充大小
        S: 步长

"""
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from pyexpat import features


# 定义函数，完成图像加载，卷积，特征图可视化
def conv_layer():
    # 1. 加载图片
    img = plt.imread('./data/universe.jpg')

    #print(f'img: {img}, img.shape: {img.shape}')

    # 2. 把图像转换为张量，从HWC 转换成 CHW
    img = torch.tensor(img, dtype=torch.float)
    img = img.permute(2, 0, 1)
    # print(f'img: {img}, img.shape: {img.shape}')

    # 3. 因为只有一张图，所以增加维度从 CHW 转换成 NCHW (1,C,H,W) 一张CHW图
    img = img.unsqueeze(0)
    # print(f'img: {img}, img.shape: {img.shape}')

    # 4. 创建卷积层
    # 参数：输入通道数，输出通道数(神经元个数)，卷积核大小，步长，填充大小
    conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)

    # 5. 卷积层前向传播
    out = conv(img)
    # print(f'out: {out}, out.shape: {out.shape}')

    # 6. 绘制4个特征图
    img4 = out[0]
    # print(f'img4: {img4}, img4.shape: {img4.shape}')

    # 7. 从CHW 转换成 HWC
    img5 = img4.permute(1, 2, 0)
    print(f'img5: {img5}, img5.shape: {img5.shape}')

    # 8. 绘制
    feature1 = img5[:,:,3].detach().numpy()
    plt.imshow(feature1)
    plt.show()




if __name__ == '__main__':
    conv_layer()