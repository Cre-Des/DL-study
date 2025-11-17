# 张量的操作与基本运算
## 1. 张量和Numpy之间相互转换

(1) 张量转换为numpy数组
```python
data_tensor.numpy()
data_tensor.numpy().copy()
```
(2) numpy转换为张量
```python
torch.from_numpy(data_numpy)
torch.tensor(data_numpy)
```
(3) 标量张量和数字转换
```python
tensor.item()
```
## 2. 张量的基本运算
### 加法
各元素相加, 如另一元素为数字, 则将该数字视为同形状且值均为该数字的张量.
```python
add_result = tensor1 + tensor2
add_result_num = tensor1 + number
add_result_num2 = tensor1.add(number)

# 修改原数据
tensor_inplace.add_(5)
tensor_inplace += 5
```

### 减法
各元素相减, 如另一元素为数字, 则将该数字视为同形状且值均为该数字的张量.
```python
sub_result = tensor1 - tensor2
sub_result_num = tensor1 - number
sub_result_num2 = tensor1.sub(number)

# 修改原数据
tensor_inplace.sub_(5)
tensor_inplace -= 5
```

### 乘法
各元素相乘, 如另一元素为数字, 则将该数字视为同形状且值均为该数字的张量.
```python
mul_result = tensor1 * tensor2
mul_result_num = tensor1 * number
mul_result_num2 = tensor1.mul(number)

# 修改原数据
tensor_inplace.mul_(5)
tensor_inplace *= 5
```

### 除法
各元素相除, 如另一元素为数字, 则将该数字视为同形状且值均为该数字的张量.
```python
div_result = tensor1 / tensor2
div_result_num = tensor1 / number
div_result_num2 = tensor1.div(number)

# 修改原数据
tensor_inplace.div_(5)
tensor_inplace /= 5

# 也可用整除运算符
div_result_num = tensor1 // number
```

### 相反
各元素取反
```python
neg_result = -tensor1
neg_result2 = torch.neg(tensor1)

# 修改原数据
tensor_inplace.neg_()
```

## 3. 张量的点乘和矩阵乘法
### 点乘
点乘 (Hadamard)指的是相同形状的张量对应位置的元素相乘, 使用 `mul` 和运算符 `*` 实现. 

设有矩阵 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{2 \times 2}$ 如下: 

$$
\mathbf{A}=\begin{bmatrix} 1&2\\ 3&4 \end{bmatrix},
\mathbf{B}=\begin{bmatrix} 5&6\\ 7&8 \end{bmatrix}
$$
则 $\mathbf{A}, \mathbf{B}$ 的点积为
$$
\mathbf{A}\circ \mathbf{B} = 
\begin{bmatrix} 1\times5&2\times6\\ 3\times7&4\times8 \end{bmatrix}
=\begin{bmatrix} 5&12\\ 21&32 \end{bmatrix}
$$
```python
hadamard_result1 = tensor1 * tensor2
hadamard_result2 = torch.mul(tensor1, tensor2)
```

### 矩阵乘法 

设有矩阵 $\mathbf{A} \in \mathbb{R}^{n \times m}, \mathbf{B} \in \mathbb{R}^{m \times p}$, 则 $\mathbf{A}\mathbf{B}\in \mathbb{R}^{n \times p}$

- 运算符 `@` 用于进行两个矩阵的乘积运算
- `torch.matmul`对进行乘积运算的两矩阵形状没有限定. 对于输入的shape不同的张量, 对应的最后几个维度必须符合矩阵运算规则
- `dot()` 只用于一维张量

```python
matmul_result1 = tensor1 @ tensor2
matmul_result2 = torch.matmul(tensor1, tensor2)

# dot() 只用于一维张量
tensor3 = torch.tensor([1, 2, 3])
tensor4 = torch.tensor([4, 5, 6])
dot_result = tensor3.dot(tensor4)
```

## 4. 张量运算函数
### 需要维度参数的函数
```python
# dim = 0 表示列 dim = 1 表示行
torch.sum(tensor, dim=None) # 求和
torch.mean(tensor, dim=None) # 求平均, 需要张量内元素类型为 float
torch.max(tensor, dim=None) # 最大值
torch.min(tensor, dim=None) # 最小值
```

### 不需要维度参数的函数
```python
torch.pow(tensor, exponent) # 求幂 等价于 ** 运算符
torch.sqrt(tensor) # 开平方
torch.exp(tensor) # 指数, e^x 其中x为张量中的元素 
torch.log(tensor) # 各元素的自然对数
torch.log2(tensor) # 各元素的 log2 对数
torch.log10(tensor)  # 各元素的 log10 对数
```