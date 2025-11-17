# 张量的创建方式
```python
import torch
import numpy as np #(可选)
```
## 1. 基本方式

``` python
    torch.tensor() #常用
    torch.Tensor()
    torch.IntTensor(), torch.FloatTensor(), torch.DoubleTensor()
```

## 2. 创建全0，全1，特定形状的张量
```python
    torch.ones(size)
    torch.zeros(size) # (常用)
    torch.full(size, fill_value)
    torch.ones_like(input)
    torch.zeros_like(input)
    torch.full_like(input, fill_value)
```

## 3. 创建线性和随机张量
```python
    # 线性张量 (常用)
    torch.arrange(start=0, end, step=1) #step 表示步长
    torch.linear(start, end, steps=100) #step 表示元素个数

    # 随机种子创建 (全局的设置, 可写在 import 下面)
    initial_seed() # 根据时间戳设置随机种子
    manual_seed(seed) # 人为指定的随机种子 (常用)

    # 随机张量
    rand(size) # 均匀分布
    randn(size) #正态分布 (常用)
    randint(low=0, high=10, size) # (常用)
```

# 4. 创建特定数据类型的张量
```python
    tensor_float=torch.tensor(data, dtype = float) # 创建特定数据类型的张量

    tensor_int = tensor_float.type(torch.int) # type() (常用)

    # half()/float()/double()/int()/long()/short()
    tensor_half = tensor_float.half()
    tensor_double = tensor_float.double()
    tensor_int = tensor_float.int()
    tensor_long = tensor_float.long()
    tensor_short = tensor_float.short()
    tensor_float_again = tensor_int.float()

```