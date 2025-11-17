'''
This is a example code file for creating tensors.

Basic ways to create tensors using PyTorch.
    torch.tensor() - Create a tensor from data.
    torch.Tensor() - Create a tensor for a special size or from data .
    torch.IntTensor(), torch.FloatTensor, torch.DoubleTensor - create a tensor for special data types.

data types:
    torch float16 - 16-bit floating point (torch.half)
    torch.float32 - 32-bit floating point (torch.float)torch.float32 is the default data type for tensors in PyTorch.
    torch.float64 - 64-bit floating point (torch.double)
    torch.uint8   - 8-bit unsigned integer
    torch.int8   - 8-bit integer
    torch.int16  - 16-bit integer (torch.short)
    torch.int32   - 32-bit integer (torch.int)
    torch.int64   - 64-bit integer (torch.long)

'''
# import necessary libraries
import torch
import numpy as np

# 1. Create a tensor from data using torch.tensor()
'''
    torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
    - data: data to be converted into a tensor (list, tuple, NumPy array
        or another tensor).
    - dtype: desired data type of the returned tensor.
    - device: the device on which the tensor is to be allocated (CPU or GPU).
    - requires_grad: if set to True, will record operations on the tensor for
        automatic differentiation.
    - pin_memory: if set to True, the tensor will be allocated in page-locked
        memory, which can improve transfer speeds to GPU.
'''
def create_tensor_from_data():
    # 1.1 Creaste a scalar tensor
    scalar_tensor = torch.tensor(10)
    print(f"Scalar Tensor: {scalar_tensor}")
    print(f"type: {type(scalar_tensor)}")
    print('-' * 30)

    # 1.2 Create a 2D tensor
    data_2d = [[1, 2, 3], [4, 5, 6]]
    tensor_2d = torch.tensor(data_2d)
    print(f"2D Tensor:\n{tensor_2d}")
    print(f"type: {type(tensor_2d)}")
    print('-' * 30)

    # 1.3 Create a 3D tensor
    data_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    tensor_3d = torch.tensor(data_3d)
    print(f"3D Tensor:\n{tensor_3d}")
    print(f"type: {type(tensor_3d)}")
    print('-' * 30)

    # 1.4 Create a tensor from a NumPy array
    data_np = np.random.randint(0, 10, (2, 3))
    tensor_from_np = torch.tensor(data_np, dtype=torch.float)
    print(f"Tensor from NumPy array:\n{tensor_from_np}")
    print(f"type: {type(tensor_from_np)}")
    print('-' * 30)

# 2. Create a tensor using torch.Tensor()
'''
torch.Tensor(*sizes) or torch.Tensor(data)
- *sizes: a variable number of integers defining the shape of the tensor to be created.
- data: data to be converted into a tensor (list, tuple, NumPy array
    or another tensor).
'''
def create_tensor_using_Tensor():
    # 2.1 Creaste a scalar tensor
    scalar_tensor = torch.Tensor(10)
    print(f"Scalar Tensor: {scalar_tensor}")
    print(f"type: {type(scalar_tensor)}")
    print('-' * 30)

    # 2.2 Create a 2D tensor
    data_2d = [[1, 2, 3], [4, 5, 6]]
    tensor_2d = torch.Tensor(data_2d)
    print(f"2D Tensor:\n{tensor_2d}")
    print(f"type: {type(tensor_2d)}")
    print('-' * 30)

    # 2.3 Create a tensor from a NumPy array
    data_np = np.random.randint(0, 10, (2, 3))
    tensor_from_np = torch.Tensor(data_np)
    print(f"Tensor from NumPy array:\n{tensor_from_np}")
    print(f"type: {type(tensor_from_np)}")
    print('-' * 30)

    # 2.4 Create a tensor with specific size
    size_tensor = torch.Tensor(2, 3)
    print(f"Tensor with specific size (2,3):\n{size_tensor}")
    print(f"type: {type(size_tensor)}")
    print('-' * 30)

# 3. Create tensors using torch.IsTensor(), torch.FloatTensor, torch.DoubleTensor
'''
    torch.IntTensor(data) - creates a tensor of 32-bit integers.
    torch.FloatTensor(data) - creates a tensor of 32-bit floating point numbers.
    torch.DoubleTensor(data) - creates a tensor of 64-bit floating point numbers.
'''
def create_tensor_using_special_formats():
    # 3.1 Creaste a scalar tensor
    scalar_tensor = torch.IntTensor(10)
    print(f"Scalar Tensor: {scalar_tensor}")
    print(f"type: {type(scalar_tensor)}")
    print('-' * 30)

    # 3.2 Create a 2D tensor
    data_2d = [[1, 2, 3], [4, 5, 6]]
    tensor_2d = torch.IntTensor(data_2d)
    print(f"2D Tensor:\n{tensor_2d}")
    print(f"type: {type(tensor_2d)}")
    print('-' * 30)

    # 3.3 Create a tensor from a NumPy array
    data_np = np.random.randint(0, 10, (2, 3))
    tensor_from_np = torch.IntTensor(data_np)
    print(f"Tensor from NumPy array:\n{tensor_from_np}")
    print(f"type: {type(tensor_from_np)}")
    print('-' * 30)

    # 3.4 If the data type is not matched, it will be converted
    data_int = np.random.randint(0, 10, (2, 3))
    tensor_converted = torch.FloatTensor(data_int) # default convert to float32
    print(f"Tensor from int data:\n{tensor_converted}")
    print(f"type: {type(tensor_converted)}")
    print('-' * 30)


# 4. define test function
if __name__ == "__main__":
    create_tensor_from_data()
    create_tensor_using_Tensor()
    create_tensor_using_special_formats()
    pass