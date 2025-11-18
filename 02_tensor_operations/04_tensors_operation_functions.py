'''
An example to illustrate tensor operation functions.

Important functions:
    with arguments dim(0 for column, 1 for row): 
    sum(tensor, dim=None, keepdim=False) - Returns the sum of all elements in the input tensor.
    mean(tensor, dim=None, keepdim=False) - Returns the mean of all elements in the input tensor.
    max(tensor, dim=None, keepdim=False) - Returns the maximum value of all elements in the input tensor.
    min(tensor, dim=None, keepdim=False) - Returns the minimum value of all elements in the input tensor.

    without arguments dim:
    pow(tensor, exponent) - Returns a new tensor with each element raised to the power of exponent.
    sqrt(tensor) - Returns a new tensor with the square-root of the elements of the input tensor.
    exp(tensor) - Returns a new tensor with the exponential of the elements of the input tensor.
    log(tensor) - Returns a new tensor with the natural logarithm of the elements of the input tensor.
    log2(tensor) - Returns a new tensor with the base-2 logarithm of the elements of the input tensor.
    log10(tensor) - Returns a new tensor with the base-10 logarithm of the elements of the input tensor.

useful functions for tensor computations.
    sum, max, min, mean
'''

# Import necessary libraries
import torch

# Create a sample tensor
tensor = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

# 1. Functions with dim argument
print(f"Original Tensor:\n{tensor}\n")
# Sum
sum_all = torch.sum(tensor)
print(f"Sum of all elements: {sum_all}")
sum_dim0 = torch.sum(tensor, dim=0)
print(f"Sum along columns (dim=0):\n{sum_dim0}")
sum_dim1 = torch.sum(tensor, dim=1)
print(f"Sum along rows (dim=1):\n{sum_dim1}\n")
print("=" * 40)

# Mean tips: Must be float tensor
mean_all = torch.mean(tensor)
print(f"Mean of all elements: {mean_all}")
mean_dim0 = torch.mean(tensor, dim=0)
print(f"Mean along columns (dim=0):\n{mean_dim0}")
mean_dim1 = torch.mean(tensor, dim=1)
print(f"Mean along rows (dim=1):\n{mean_dim1}\n")
print("=" * 40)

# Max
max_all = torch.max(tensor)
print(f"Max of all elements: {max_all}")
max_dim0 = torch.max(tensor, dim=0)
print(f"Max along columns (dim=0):\n{max_dim0.values}")
max_dim1 = torch.max(tensor, dim=1)
print(f"Max along rows (dim=1):\n{max_dim1.values}\n")
print("=" * 40)

# Min
min_all = torch.min(tensor)
print(f"Min of all elements: {min_all}")
min_dim0 = torch.min(tensor, dim=0)
print(f"Min along columns (dim=0):\n{min_dim0.values}")
min_dim1 = torch.min(tensor, dim=1)
print(f"Min along rows (dim=1):\n{min_dim1.values}\n")
print("=" * 40)

# 2. Functions without dim argument
# Power
print(f"Original Tensor:\n{tensor}\n")
print(f"Tensor raised to power 2:\n{torch.pow(tensor, 2)}\n")
print(f"Tensor raised to power 3:\n{torch.pow(tensor, 3)}\n")
print(f"Tensor raised to power 2 (using **):\n{tensor ** 2}\n")
print("=" * 40)

# Square Root
print(f"Square root of Tensor:\n{torch.sqrt(tensor)}\n") # each element square root
print("=" * 40)

# Exponential e^{x}
print(f"Exponential of Tensor:\n{torch.exp(tensor)}\n") # each element exponential
print("=" * 40)

# Natural Logarithm
print(f"Natural Logarithm of Tensor:\n{torch.log(tensor)}\n") # each element natural log
print("=" * 40)

# Base-2 Logarithm
print(f"Base-2 Logarithm of Tensor:\n{torch.log2(tensor)}\n") # each element base-2 log
print("=" * 40)

# Base-10 Logarithm
print(f"Base-10 Logarithm of Tensor:\n{torch.log10(tensor)}\n") # each element base-10 log
print("=" * 40)
