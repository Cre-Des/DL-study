"""
基础图像操作

涉及函数
img = plt.imread('./data/universe.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# 1. 绘制全黑全白图
def draw_black_white():
    # 生成全黑图片 (0-255)
    # HWC H: 高度 W:宽度 C:通道数
    img1 = np.zeros([200, 200, 3])
    print(f'img1: {img1}')
    plt.imshow(img1)
    plt.show()

    # 绘制全白图片
    img2 = torch.full([200, 200, 3], 255)
    print(f'img2: {img2}')
    plt.imshow(img2)
    plt.show()

# 2. 加载图片
def load_image():
    # 加载图片
    img = plt.imread('./data/universe.jpg')
    print(f'img: {img}')
    print(f'img.shape: {img.shape}')

    # 保存图片
    # plt.imsave('./data/universe_copy.jpg', img)

    # 显示图片
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # draw_black_white()
    load_image()