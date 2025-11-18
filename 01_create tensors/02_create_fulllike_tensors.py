'''
An example to create tensors using torch.ones, torch.zeros() and torch.full_like()  functions.

functions.
  - torch.ones(size, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  - torch.zeros(size, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  - torch.full(size, fill_value, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  - torch.ones_like(input, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  - torch.zeros_like(input, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  - torch.full_like(input, fill_value, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

important functions to create tensors filled with specific values.
    torch.zeros(), torch.full()
'''

# Import necessary libraries
import torch

# 1. Create a tensor filled with ones using torch.ones()
# 1.1 Create a 2*3 tensor filled with ones
tensor_ones = torch.ones(2, 3)
print(f"Tensor filled with ones:\n{tensor_ones}")
print(f"type: {type(tensor_ones)}")
print('-' * 30)

# 1.2 Create a 3*2 tensor using torch.ones_like()
tensor_base = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensor_ones_like = torch.ones_like(tensor_base)
print(f"Tensor filled with ones (like another tensor):\n{tensor_ones_like}")
print(f"type: {type(tensor_ones_like)}")
print('-' * 30)

# 2. Create a tensor filled with zeros using torch.zeros()
# 2.1 Create a 2*3 tensor filled with zeros
tensor_zeros = torch.zeros(2, 3)
print(f"Tensor filled with zeros:\n{tensor_zeros}")
print(f"type: {type(tensor_zeros)}")
print('-' * 30)

# 2.2 Create a 3*2 tensor using torch.zeros_like()
tensor_zeros_like = torch.zeros_like(tensor_base)
print(f"Tensor filled with zeros (like another tensor):\n{tensor_zeros_like}")
print(f"type: {type(tensor_zeros_like)}")
print('-' * 30)

# 3. Create a tensor filled with a specific value using torch.full()
# 3.1 Create a 2*3 tensor filled with the value 7
tensor_full = torch.full((2, 3), 7)
print(f"Tensor filled with 7:\n{tensor_full}")
print(f"type: {type(tensor_full)}")
print('-' * 30)

# 3.2 Create a 3*2 tensor using torch.full_like()
tensor_full_like = torch.full_like(tensor_base, 9)
print(f"Tensor filled with 9 (like another tensor):\n{tensor_full_like}")
print(f"type: {type(tensor_full_like)}")
print('-' * 30)



