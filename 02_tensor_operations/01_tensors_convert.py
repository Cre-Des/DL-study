'''
An example to illustrate tensor conversion operations.

Tensor to Numpy array.
    - tensor.numpy() Shared memory.
    - tensor.numpy().copy() No shared memory.

Numpy array to Tensor.
    - torch.from_numpy(array) Shared memory.
    - torch.tensor(array) No shared memory.

Get information from Scalar Tensor.
    - tensor.item()

Important functions:
    - tensor.numpy()
    - torcj.tensor()
    - tensor.item()
'''

# Import necessary libraries
import torch
import numpy as np

# 1. Tensor to Numpy array
def tensor_to_numpy():
    # Create a sample tensor
    tensor = torch.tensor([1, 2, 3, 4, 5])
    print(f"Tensor : {tensor}, Type: {type(tensor)}")

    # Convert to Numpy array
    numpy_array_shared = tensor.numpy()  # Shared memory
    print(f"Numpy Array (Shared Memory) : {numpy_array_shared}, Type: {type(numpy_array_shared)}")
    numpy_array_no_shared = tensor.numpy().copy()  # No shared memory
    print(f"Numpy Array (No Shared Memory) : {numpy_array_no_shared}, Type: {type(numpy_array_no_shared)}")

    # Modify the numpy array
    numpy_array_shared[0] = 10
    print(f"Modified Numpy Array (Shared Memory) : {numpy_array_shared}")
    print(f"Tensor after modifying Numpy Array (Shared Memory) : {tensor}")

    numpy_array_no_shared[1] = 20
    print(f"Modified Numpy Array (No Shared Memory) : {numpy_array_no_shared}")
    print(f"Tensor after modifying Numpy Array (No Shared Memory) : {tensor}")
    print("=" * 40)

# 2. Numpy array to Tensor
def numpy_to_tensor():
    # Create a sample numpy array
    numpy_array = np.array([1, 2, 3, 4, 5])
    print(f"Numpy Array : {numpy_array}, Type: {type(numpy_array)}")

    # Convert to Tensor
    tensor_shared = torch.from_numpy(numpy_array)  # Shared memory
    print(f"Tensor (Shared Memory) : {tensor_shared}, Type: {type(tensor_shared)}")
    tensor_no_shared = torch.tensor(numpy_array)  # No shared memory
    print(f"Tensor (No Shared Memory) : {tensor_no_shared}, Type: {type(tensor_no_shared)}")

    # Modify the tensor
    tensor_shared[0] = 10
    print(f"Modified Tensor (Shared Memory) : {tensor_shared}")
    print(f"Numpy Array after modifying Tensor (Shared Memory) : {numpy_array}")

    tensor_no_shared[1] = 20
    print(f"Modified Tensor (No Shared Memory) : {tensor_no_shared}")
    print(f"Numpy Array after modifying Tensor (No Shared Memory) : {numpy_array}")
    print("=" * 40)

# 3. Get information from Scalar Tensor
def scalar_tensor_info():
    # Create a scalar tensor
    scalar_tensor = torch.tensor(42)

    # Matrix_tensor = torch.tensor([[1, 2], [3, 4]])
    print(f"Scalar Tensor : {scalar_tensor}, Type: {type(scalar_tensor)}")

    # Get the value as a Python number
    value = scalar_tensor.item()
    # value_matrix = Matrix_tensor.item() # This will raise an error since Matrix_tensor is not a scalar (a Tensor with 4 elements cannot be converted to Scalar)
    
    print(f"Value from Scalar Tensor : {value}, Type: {type(value)}")
    print("=" * 40)

# Test the functions
if __name__ == "__main__":
    tensor_to_numpy()
    numpy_to_tensor()
    scalar_tensor_info()
