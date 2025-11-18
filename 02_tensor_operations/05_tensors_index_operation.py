'''
A example of tensor indexing operations using PyTorch.

Classifications:
    - Simple column and row indexing (important)
    - list-based indexing
    - range-based indexing (important)
    - boolean indexing
    - multi-dimensional indexing (important)
'''

# Import necessary libraries
import torch

# set random seed for reproducibility
torch.manual_seed(24)

# Create a 5x5 tensor with random values
tensor = torch.randint(1,10,(5, 5))
print("Original Tensor:\n", tensor)
print('='*50)

# 1. Simple column and row indexing (tensor[row, col])
# 1.1 Accessing a specific row
row_index = 2
specific_row = tensor[row_index]
print(f"\nRow at index {row_index}:\n", specific_row)

specific_row_alternate = tensor[row_index, :]
print(f"\nRow at index {row_index} (alternate method):\n", specific_row_alternate)

# 1.2 Accessing a specific column
col_index = 3
specific_column = tensor[:, col_index]
print(f"\nColumn at index {col_index}:\n", specific_column)

print('='*50)

# 2. List-based indexing (tensor[[row_indices], [col_indices]])
# 2.1 Accessing (0,1), (1,2) elements
print(f'\nElements at (0,1) and (1,2):\n', tensor[[0, 1], [1, 2]])

# 2.2 Accessing (1,2), (3,4) elements
print(f'\nElements at (1,2), (3,4):\n', tensor[[1, 3], [2, 4]]) 

# 2.3 Accessing 0th , 1st row's 1st and 2nd columns
print(f'\n0th and 1st rows, 1st and 2nd columns:\n', tensor[[[0], [1]], [1, 2]])

print('='*50)

# 3. Range-based indexing
# 3.1 Accessing first 3 rows and first 2 columns
print(f'\nFirst 3 rows and first 2 columns:\n', tensor[:3, :2])

# 3.2 Accessing from 2nd rows to last row and first 2 columns
print(f'\nFrom 2nd row to last row, first 2 columns:\n', tensor[1:, :2])

# 3.3 Accessing all odd indexed rows and all even indexed columns
print(f'\nAll even indexed rows and all odd indexed columns:\n', tensor[::2, 1::2])

print('='*50)

# 4. Boolean indexing
# 4.1 3rd column greater than 5
print(f'\nElements in 3rd column greater than 5:\n', tensor[tensor[:, 2] > 5, 2])

# 4.2 The row of the elements in 3rd column greater than 5
print(f'\nRows where 3rd column elements are greater than 5:\n', tensor[tensor[:, 2] > 5])

print('='*50)

# 5. Multi-dimensional indexing
# creating a 3D tensor
tensor_3d = torch.randint(1,10,(2, 3, 4))
print("\n3D Tensor:\n", tensor_3d)

# 5.1 Accessing 1st element in 0th dimension.
print(f'\n1st element in 0th dimension:\n', tensor_3d[0, :, :])

# 5.2 Accessing 1st element in 1st dimension.
print(f'\n1st element in 1st dimension:\n', tensor_3d[:, 0, :])

# 5.3 Accessing 1st element in 2nd dimension.
print(f'\n2nd element in 2nd dimension:\n', tensor_3d[:, :, 0])