'''
An example of tensor concatenation operations using PyTorch.

Important functions:
    - torch.cat() - concatenate tensors along a specified dimension (without changing their number of dimensions)
    - torch.stack() - stack tensors along a new dimension (changing their number of dimensions)
'''

# Import necessary libraries
import torch

# create random tensors
tensor1 = torch.randint(1, 10, (2, 3))
tensor2 = torch.randint(1, 10, (2, 3))
tensor3 = torch.randint(1, 10, (5, 3))
print("Tensor 1:\n", tensor1)
print(f'tensor1 shape: {tensor1.shape}')
print("Tensor 2:\n", tensor2)
print(f'tensor2 shape: {tensor2.shape}')
print('='*50)

# 1. Using torch.cat() to concatenate tensors along different dimensions

# Concat requires that the tensors have the same shape, except in the dimension being concatenated.
# 1.1 Concatenate along dimension 0 (rows)
tensor_cat_dim0 = torch.cat((tensor1, tensor2), dim=0)
print("Concatenated along dim 0:\n", tensor_cat_dim0)
print(f'tensor_cat_dim0 shape: {tensor_cat_dim0.shape}')
print('-'*50)

# 1.2 Concatenate along dimension 1 (columns)
tensor_cat_dim1 = torch.cat((tensor1, tensor2), dim=1)
print("Concatenated along dim 1:\n", tensor_cat_dim1)
print(f'tensor_cat_dim1 shape: {tensor_cat_dim1.shape}')
print('='*50)

# 2. Concatenating tensors of different sizes along a specific dimension
# 2.1 Concatenate tensor1 and tensor3 along dimension 0
tensor_cat_dim0_3 = torch.cat((tensor1, tensor3), dim=0)
print("Concatenated tensor1 and tensor3 along dim 0:\n", tensor_cat_dim0_3)
print(f'tensor_cat_dim0_3 shape: {tensor_cat_dim0_3.shape}')
print('-'*50)

# 2.2 Attempt to concatenate tensor1 and tensor3 along dimension 1 (will raise an error)

# Uncommenting the following lines will raise a RuntimeError:
# Sizes of tensors must match except in dimension 1. Expected size 2 but got size 5 for tensor number 1 in the list.

# tensor_cat_dim1_3 = torch.cat((tensor1, tensor3), dim=1)
# print("Concatenated tensor1 and tensor3 along dim 1:\n", tensor_cat_dim1_3)
# print(f'tensor_cat_dim1_3 shape: {tensor_cat_dim1_3.shape }')
# print('='*50)

# 3. Using torch.stack() to stack tensors along a new dimension
# Stack requires that the tensors have the same shape.

# 3.1 Stack tensor1 and tensor2 along dimension 0
tensor_stack_dim0 = torch.stack((tensor1, tensor2), dim=0)
print("Stacked along dim 0:\n", tensor_stack_dim0)
print(f'tensor_stack_dim0 shape: {tensor_stack_dim0.shape}')
print('-'*50)

# 3.2 Stack tensor1 and tensor2 along dimension 1
tensor_stack_dim1 = torch.stack((tensor1, tensor2), dim=1)
print("Stacked along dim 1:\n", tensor_stack_dim1)
print(f'tensor_stack_dim1 shape: {tensor_stack_dim1.shape}')
print('='*50)



