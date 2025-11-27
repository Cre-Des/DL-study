# RNN
RNN 会逐个读取句子中的词语，并在每一步结合当前词和前面的上下文信息，不断更新对句子的理解。通过这种机制，RNN 能够持续建模上下文，从而更准确地把握句子的整体语义。因此RNN曾是序列建模领域的主流模型，被广泛应用于各类NLP任务。

## 基础结构
RNN（循环神经网络）的核心结构是**一个具有循环连接的隐藏层**，它以**时间步**（time step）为单位，依次处理输入序列中的每个 token。
在每个时间步，RNN 接收**当前 token 的向量**和**上一个时间步的隐藏状态**（即隐藏层的输出），计算并**生成新的隐藏状态**，并将其传递到下一时间步。

**1. 计算隐藏状态：** 每个时间步的隐藏状态 $h_t$ 是根据当前输入 $x_t$ 和前一时刻的隐藏状态 $h_{t-1}$ 计算的。
$$h_t = \tanh(W_{ih}x_t + b_{ih} + W_{hh}h_{t-1} + b_{hh})$$
上述公式中:
- $W_{ih}$ 表示输入数据的权重
- $b_{ih}​$ 表示输入数据的偏置
- $W_{hh}​$ 表示输入隐藏状态的权重
- $b_{hh}$ 表示输入隐藏状态的偏置
- $h_{t-1}$ 表示输入隐藏状态
- $h_t​$ 表示输出隐藏状态

**2. 计算当前时刻的输出：** 网络的输出 $y_t$ 是当前时刻的隐藏状态经过一个线性变换得到的。
$$y_t=W_{hy}h_t+b_y​$$
- $y_t​$ 是当前时刻的输出（通常是一个向量，表示当前时刻的预测值，RNN层的预测值）
- $h_t​$ 是当前时刻的隐藏状态
- $W_{hy}$ 是从隐藏状态到输出的权重矩阵
- $b_y$ 是输出层的偏置项

**3. 词汇表映射**：输出 $y_t$ 是一个**向量**，该向量经过**全连接层**后输出得到最终预测结果 $y_{pred}$，$y_{pred}$中每个元素代表当前时刻生成词汇表中某个词的得分（或概率，通过激活函数：如**softmax**）。词汇表有多少个词，$y_{pred}$就有多少个元素值，**最大元素值对应的词就是当前时刻预测生成的词**。

## 多层结构
为了让模型捕捉更复杂的语言特征，可以将**多个 RNN 层按层次堆叠**起来，使不同层学习不同层次的语义信息。  
这种设计的核心假设是：底层网络更容易捕捉**局部模式**（如词组、短语），而高层网络则能学习**更抽象的语义信息**（如句子主题或语境）。  
多层RNN结构中，每一层的 **输出序列** 会作为下一层的 **输入序列**，最底层RNN接收原始输入序列，顶层 RNN的输出作为最终结果用于后续任务。

## 双向结构
基础的 RNN 在每个时间步只输出一个隐藏状态，该状态仅包含来自上文的信息，而无法利用当前词之后的下文。

使用**双向 RNN**（Bidirectional RNN），模型可以在每个时间步同时利用前文和后文的信息，从而获得更全面的上下文表示，有助于提升序列标注等任务的预测效果。

双向RNN同时使用两层 RNN：  
**正向 RNN**：按照时间顺序（从前到后）处理序列；  
**反向 RNN**：按照逆时间顺序（从后到前）处理序列。  
每个时间步的输出，是正向和反向隐藏状态的组合（例如拼接或求和）。

## 多层+双向结构
多层结构和双向结构还可组合使用，每层都是一个双向RNN。

## API使用
PyTorch 提供了 `torch.nn.RNN` 模块用于构建循环神经网络（Recurrent Neural Network, RNN）。该模块支持单层或多层结构，也可通过设置参数启用双向 RNN（bidirectional），适用于处理序列建模相关任务。

```python
torch.nn.RNN(
    input_size,             # 每个时间步输入特征的维度（词向量维度）
    hidden_size,            # 隐藏状态的维度
    num_layers=1,           # RNN层数
    nonlinearity="tanh",    # 激活函数 （tanh 或 ReLU）
    bias=True,              # 是否使用偏置项
    batch_first=False,      # 输入张量是否是 (batch, seq, feature)，默认 False 表示 (seq, batch, feature)
    dropout=0.0,            # 除最后一层外，其余层之间的 dropout 概率
    bidirectional=False,    # 是否使用双向 RNN
    device=None,            # 运行设备
    dtype=None,             # 运行数据类型
)
```

```python
rnn = torch.nn.RNN()
output, h_n = rnn(input, h_0)
```

输入 
- `input`	  
输入序列，形状为(seq_len, batch_size, input_size)，如果 `batch_first=True`，则为 (batch_size, seq_len, input_size)

- `h_0`	  
可选，初始隐藏状态，形状为 (num_layers × num_directions, batch_size, hidden_size)

输出	
- `output`	  
RNN层的输出，包含最后一层每个时间步的隐藏状态，形状为 (seq_len, batch_size, num_directions × hidden_size )，如果如果 `batch_first=True`，则为(batch_size, seq_len, num_directions × hidden_size )

- `h_n`	  
最后一个时间步的隐藏状态，包含每一层的每个方向，形状为 (num_layers × num_directions, batch_size, hidden_size)



4.1.8 存在问题
