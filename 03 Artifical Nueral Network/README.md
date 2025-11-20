# ANN 人工神经网络

## 1. 神经网络基础
人工神经网络（Artificial Neural Network， 简写为**ANN**）也简称为神经网络（NN），是一种模仿生物神经网络结构和功能的**计算模型**。它由多个互相连接的人工神经元（也称为节点）构成，可以用于处理和学习复杂的数据模式，尤其适合解决非线性问题。人工神经网络是机器学习中的一个重要模型，尤其在深度学习领域中得到了广泛应用。接收数据是二维的

###  如何构建神经网络
神经网络是由多个神经元组成，构建神经网络就是在构建神经元。

神经网络中信息只向一个方向移动，即从输入节点向前移动，通过隐藏节点，再向输出节点移动。其中的基本部分是:

1. **输入层（Input Layer）**: 即输入x的那一层（如图像、文本、声音等）。每个输入特征对应一个神经元。输入层将数据传递给下一层的神经元。
2. **输出层（Output Layer）**: 即输出y的那一层。输出层的神经元根据网络的任务（回归、分类等）生成最终的预测结果。
3. **隐藏层（Hidden Layers）**: 输入层和输出层之间都是隐藏层，神经网络的“深度”通常由隐藏层的数量决定。隐藏层的神经元通过加权和激活函数处理输入，并将结果传递到下一层。

特点: 
- 同一层的神经元之间没有连接
- 第N层的每个神经元和第N-1层的所有神经元相连（这就是Fully Connected的含义)，这就是**全连接神经网络（FCNN）**
- 全连接神经网络接收的样本数据是**二维的**，数据在每一层之间需要以二维的形式传递
- 第N-1层神经元的输出就是第N层神经元的输入
- 每个连接都有一个权重值（ $\boldsymbol{w}$ 系数和 $\boldsymbol{b}$ 系数）

###  神经网络内部状态值和激活值
每一个神经元工作时，**前向传播**会产生两个值，**内部状态值（加权求和值）**和**激活值**；**反向传播**时会产生**激活值梯度**和**内部状态值梯度**。

**内部状态值** :神经元或隐藏单元的内部存储值，它反映了当前神经元接收到的输入、历史信息以及网络内部的权重计算结果。
$$\boldsymbol{z} = \mathbf{W}\boldsymbol{x} + \boldsymbol{b}$$

**激活值** : 通过激活函数（如 ReLU、Sigmoid、Tanh）对内部状态值进行非线性变换后得到的结果。激活值决定了当前神经元的输出。
$$a = f(\boldsymbol{z})$$

## 2. 激活函数

激活函数用于对每层的输出数据进行变换，进而为整个网络注入了 **非线性因素**。此时，神经网络就可以
拟合各种曲线。

1.没有引入非线性因素的网络等价于使用一个线性模型来拟合  
2.通过给网络输出增加激活函数，实现引入非线性因素，使得网络模型可以逼近任意函数，提升网络对
复杂问题的拟合能力.

### sigmoid

公式: 
$$\frac{1}{1+e^{-x}}$$

导数:
$$f'(x) = \frac{1}{1+e^{-x}}\left(1-\frac{1}{1+e^{-x}}\right)=f(x)(1-f(x))$$

在 $[-6,6]$ 有效果， 在$[-3, 3]$ 效果明显.导数结果在 $[0, 0.25]$ 之间, 值分布于 $[-6, 6]$之间.  

![sigmoid.png](fig/sigmoid.png)

sigmoid函数可以将任意的输入映射到(0，1)之间，当输入的值大致在$<6$或者$>6$时，意味着输入任何值得
到的激活值都是差不多的，这样会丢失部分的信息。  

一般来说，sigmoid网络在 **5层之内** 就会产生 **梯度消失** 现象。而且，该激活函数并不是以0为中心（以0.5为中心）的，所以在
实践中这种激活函数使用的很少。sigmoid函数一般只用于二分类的 **输出层**。

### Tanh

公式: 
$$\frac{1- e^{-2x}}{1+e^{-2x}}$$

导数:
$$f'(x) = 1-f^2(x)$$

在 $[-3,3]$ 有效果， 在$[-1, 1]$ 效果明显  
![Tanh.png](fig/Tanh.png)

Tanh函数将输入映射到（1，1）之间，图像以0为中心，在0点对称，当输入大概$<-3$
或者$>3$时将被映射为$-1$或者$1$。其导数值范围（0，1），当输入的值大概$<-3$或者$>3$
时，其导数近似0。  
与Sigmoid相比，它是以0为中心的，且梯度相对于Sigmoid大，使得其 **收敛速度要比
Sigmoid快**，减少迭代次数。然而，从图中可以看出，Tanh两侧的导数也为0，同样会造
成梯度消失。  
若使用时可在 **隐藏层** 使用tanh函数，在 **输出层** 使用sigmoid函数。

### ReLU
默认情况下， ReLU只考虑正样本。  
公式: 
$$f(x) = \max (0,x)$$

导数:  
$$f'(x) = 0\ or\ 1$$

![ReLU .png](fig/ReLU.png)

ReLU激活函数将小于0的值映射为0，而大于0的值则保持不变，它更加重视正信号，
而忽略负信号，这种激活函数运算更为简单，能够提高模型的训练效率。  
当 $x<O$ 时，ReLU导数为0，而当 $x>O$ 时，则不存在饱和问题。
所以，ReLU能够在 $x>O$ 时保持梯度不衰减，从而缓解梯度消失问题。
然而，随着训练的推进，部分输入会落入小于0区域，导致对应权重无法更新。
这种现象被称为“神经元死亡”  
ReLU是目前最常用的激活函数。

- 与sigmoid相比，ReLU的优势是： 采用sigmoid函数，计算量大（指数运算），反向传播求误差梯度时，计算量相对大，而采用
Relu激活函数，整个过程的计算量节省很多。
- sigmoid函数反向传播时，很容易就会出现梯度
消失的情况，从而无法完成深层网络的训练。Relu会使一部分神经元的输出为0，这样就造成
了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。

### softmax
softmax用于多分类过程中，它是二分类函数sigmoid在多分类上的推广，目的是将多分类的结果以概率的形式展现出来。
公式: 
$$f(x) = softmax(z_i) \frac{e^{z_i}}{\sum_j e^{z_j}}$$

### 激活函数的选择
**隐藏层** 常用的激活函数(从左到右优先级递减)
ReLU, LeakyReLU, PReLU, Tanh, Sigmoid(少用)
如果你使用了ReLU 需要注意Dead ReLU问题, 避免出现0梯度而导致过多的神经元死亡

**输出层** 常用的激活函数
二分类选择 sigmoid  
多分类选择 Softmax  
回归问题选择 identity

## 3. 参数初始化

我们在构建网络之后，网络中的参数是需要初始化的。我们需要初始化的参数主要有**权重**和**偏置**，**偏置一般初始化为0即可**，而对权重的初始化则会更加重要。

### 参数初始化的作用

- **防止梯度消失或爆炸**：初始权重值过大或过小会导致梯度在反向传播中指数级增大或缩小。
- **提高收敛速度**：合理的初始化使得网络的激活值分布适中，有助于梯度高效更新。
- **保持对称性破除**：权重的初始化需要打破对称性，否则网络的学习能力会受到限制。


### 无法打破对称性的初始化
- 全0初始化(记忆)
    `nn.init.zeros_()`  
- 全1初始化
    `nn.init.ones_()`  
- 固定值初始化
    `nn.init.constant_()`  

### 可以打破对称性的初始化
- 随机初始化
    `nn.init.uniform_()`
    `nn.init.normal_()`

- kaiming 初始化(+ ReLU) 专为ReLU和其变体设计，考虑到ReLU激活函数的特性，对输入维度进行缩放(记忆)  
    正态分布的 he初始化
        `nn.init.kaiming_normal_()`  
    均匀分布的 he初始化
        `nn.init.kaiming_uniform_()`  
- xavier初始化  适用于Sigmoid、Tanh 等激活函数，解决梯度消失问题(记忆)  
    正态分布的 xavier初始化
        `nn.init.xavier_normal_()`  
    均匀分布的 xavier初始化
        `nn.init.xavier_uniform_()`

激活函数 **ReLU 及其系列**，优先 kaiming 初始化  
激活函数 **非 ReLU**，优先 xavier初始化

若为 **浅层网络**，可以考虑随机初始化  
若为 **深层网络**，kaiming 初始化和xavier初始化

## 4. 神经网络搭建和参数计算
在pytorch中定义深度神经网络其实就是层堆叠的过程，继承自nn.Module，实现两个方法：

- `__init__`方法中定义网络中的层结构，主要是全连接层，并进行初始化
- forward方法，在调用神经网络模型对象的时候，底层会自动调用该函数。该函数中为初始化定义的layer传入数据，进行前向传播等。
  
深度学习的4个步骤：
1. 准备数据.
2. 搭建神经网络
3. 模型训练
4. 模型测试

神经网络搭建流程：  
1. 定义一个类，继承:`nn.Module`。  
2. 在`__init__()`方法中，搭建神经网络。  
3. 在`forward()`方法中，完成前向传播。  

![nn_create](fig/nn_create.png)

- 第1个隐藏层：权重初始化采用标准化的xavier初始化 激活函数使用sigmoid
- 第2个隐藏层：权重初始化采用标准化的He初始化 激活函数采用relu
- out输出层线性层 假若多分类，采用softmax做数据归一化

### 构造神经网络模型代码

```python
import torch
import torch.nn as nn
from torchsummary import summary  # 计算模型参数,查看模型结构, pip install torchsummary -i https://mirrors.aliyun.com/pypi/simple/


# 创建神经网络模型类
class Model(nn.Module):
    # 初始化属性值
    def __init__(self):
        # 调用父类的初始化属性值，确保nn.Module的初始化代码能够正确执行
        super(Model, self).__init__()
        # 创建第一个隐藏层模型, 3个输入特征,3个输出特征
        self.linear1 = nn.Linear(3, 3)
        # 初始化权重
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 创建第二个隐藏层模型, 3个输入特征(上一层的输出特征),2个输出特征
        self.linear2 = nn.Linear(3, 2)
        # 初始化权重
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear2.bias)
        # 创建输出层模型
        self.out = nn.Linear(2, 2)

	# 创建前向传播方法, 调用神经网络模型对象时自动执行forward()方法
    def forward(self, x):
        # 数据经过第一个线性层
        x = self.linear1(x)
        # 使用sigmoid激活函数
        x = torch.sigmoid(x)

        # 数据经过第二个线性层
        x = self.linear2(x)
        # 使用relu激活函数
        x = torch.relu(x)

        # 数据经过输出层
        x = self.out(x)
        # 使用softmax激活函数
        # dim=-1:每一维度行数据相加为1
        x = torch.softmax(x, dim=-1)

        return x
```

### 训练神经网络模型代码

```python
# 创建构造模型函数
def train():
    # 实例化model对象
    my_model = Model()

    # 随机产生数据
    my_data = torch.randn(5, 3)
    print("my_data-->", my_data)
    print("my_data shape", my_data.shape)

    # 数据经过神经网络模型训练
    output = my_model(my_data)
    print("output-->", output)
    print("output shape-->", output.shape)

    # 计算模型参数
    # 计算每层每个神经元的w和b个数总和
    print("======计算模型参数======")
    summary(my_model, input_size=(3,), batch_size=5)
    
    # 查看模型参数
    print("======查看模型参数w和b======")
    for name, parameter in my_model.named_parameters():
        print(name, parameter)


if __name__ == '__main__':
    train()
```

###  查看模型参数

通常继承nn.Module，撰写自己的网络层。它强大的封装不需要我们定义可学习的参数（比如卷积核的权重和偏置参数）。

查看封装好的可学习的网络参数

  - `模块实例名.name_parameters()`,会分别返回`name`和`parameter`

  ```python
  # 实例化model对象
  mymodel = Model()
  
  # 查看网络参数
  for name, parameter in mymodel.named_parameters():
      # print('name--->', name)
      # print('parameter--->', parameter)
      print(name, parameter)
  ```

### 模型参数的计算

以第一个隐层为例：该隐层有3个神经元，每个神经元的参数为：4个（w1,w2,w3,b1），所以一共用3x4=12个参数。 


  ```python
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Linear-1                     [5, 3]              12
              Linear-2                     [5, 2]               8
              Linear-3                     [5, 2]               6
  ================================================================
  Total params: 26
  Trainable params: 26
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.00
  Forward/backward pass size (MB): 0.00
  Params size (MB): 0.00
  Estimated Total Size (MB): 0.00
  ----------------------------------------------------------------
  ```
