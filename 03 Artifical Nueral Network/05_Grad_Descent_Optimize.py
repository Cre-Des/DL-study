"""
梯度下降优化方法

- 指数移动加权平均
    近30天分布情况，β 越大，越平缓

- 动量算法Momentum (常用)
    s_t=β s_{t−1} + (1−β) g_t
    w_t=w_{t−1}−ηs_t

- AdaGrad 自适应学习率
    s_t = s_{t-1} + g_t ⊙ g_t
        s_t 累计平方梯度  s_{t-1}历史累计平方梯度  g_t 本次梯度
    学习率 = 学习率/(sqrt{s_t}+σ) 调整学习率
        σ = 1e-10 防止分母变零
    w_t = w_{t-1} - 学习率 * g_t

- RMSProp 自适应学习率
    s_t = β s_{t-1} + (1−β) g_t ⊙ g_t
        s_t 累计平方梯度  s_{t-1}历史累计平方梯度  g_t 本次梯度
    学习率 = 学习率/(sqrt{s_t}+σ) 调整学习率
        σ = 1e-10 防止分母变零
    w_t = w_{t-1} - 学习率 * g_t

- Adam 自适应矩估计 (常用)
    既优化学习率，又优化梯度。
    一阶矩：
        m_t=β m_{t−1} + (1−β) g_t   梯度
        s_t = β s_{t-1} + (1−β) g_t ⊙ g_t 学习率
    二阶矩：
        mt^ = mt / 1 - β1^t
        st^ = st / 1 - β2^t
    w_t = w_{t-1} - 学习率/(sqrt{st^}+σ) * mt^
"""
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

ELEMENT_NUMBER = 30

# 1. 实际平均温度
def real_mean_temperature():
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
    print(temperature)
    # 绘制平均温度
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, temperature, color='r')
    plt.scatter(days, temperature)
    plt.show()


# 2. 指数加权平均温度
def exponential_weighted_moving_average(beta=0.9):
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
    print(temperature)

    exp_weight_avg = []
    # idx从1开始
    for idx, temp in enumerate(temperature, 1):
        # 第一个元素的 EWA 值等于自身
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        # 第二个元素的 EWA 值等于上一个 EWA 乘以 β + 当前气温乘以 (1-β)
        # idx-2：2-2=0，exp_weight_avg列表中第一个值的下标值
        new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)

    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, exp_weight_avg, color='r')
    plt.scatter(days, temperature)
    plt.show()

# 3. 动量法Momentum
def momentum_weighted_moving_average(beta=0.9):
    # 1. 初始化权重参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)
    # 2. 定义损失函数
    criterion = (w ** 2)/ 2.0

    # 3. 创建优化器 SGD（随机梯度下降），加入参数momentum，就是动量法
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)

    # 4. 计算梯度值
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w}, w.grad:{w.grad}')
    # 5 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    criterion = (w ** 2)/ 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w}, w.grad:{w.grad}')

# 4. AdaGrad
def ada_grad_weighted_moving_average(beta=0.9):
    # 1. 初始化权重参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)
    # 2. 定义损失函数
    criterion = (w ** 2)/ 2.0

    # 3. 创建优化器
    optimizer = optim.Adagrad(params=[w], lr=0.01)

    # 4. 计算梯度值
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w}, w.grad:{w.grad}')
    # 5 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    criterion = (w ** 2)/ 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w}, w.grad:{w.grad}')

# 5. RMSProp
def rms_prop_weighted_moving_average(beta=0.9):
    # 1. 初始化权重参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)
    # 2. 定义损失函数
    criterion = (w ** 2)/ 2.0

    # 3. 创建优化器
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.99)

    # 4. 计算梯度值
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w}, w.grad:{w.grad}')
    # 5 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    criterion = (w ** 2)/ 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w}, w.grad:{w.grad}')

# 6. Adam
def adam_weighted_moving_average(beta=0.9):
    # 1. 初始化权重参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)
    # 2. 定义损失函数
    criterion = (w ** 2)/ 2.0

    # 3. 创建优化器
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999))  # betas = (梯度的衰减系数，学习率的衰减系数)

    # 4. 计算梯度值
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w}, w.grad:{w.grad}')
    # 5 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    criterion = (w ** 2)/ 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w}, w.grad:{w.grad}')

if __name__ == '__main__':
    # real_mean_temperature()
    # exponential_weighted_moving_average(0.9)
    # exponential_weighted_moving_average(0.4)
    # momentum_weighted_moving_average()
    # ada_grad_weighted_moving_average()
    # rms_prop_weighted_moving_average()
    adam_weighted_moving_average()
