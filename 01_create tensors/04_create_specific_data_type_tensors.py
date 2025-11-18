'''
An example to show how to create specific data type tensors and convert tensors data types in PyTorch.

functions.
    - type()
    - half()
    - float()
    - double()
    - int()
    - long()
    - short()

'''

# Import necessary libraries
import torch

# Create a tensor with specific data type
tensor_float = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float)
print(f"Original Tensor (float32):\n{tensor_float}")
print(f"elements type: {tensor_float.dtype}, type: {type(tensor_float)}")
print('-' * 30)

# Convert tensor to different data types after creation

# Convert to int (int16) (recommended to use type() method)
tensor_int = tensor_float.type(torch.int)
print(f"Tensor converted to int (int32):\n{tensor_int}")
print(f"elements type: {tensor_int.dtype}, type: {type(tensor_int)}")
print('-' * 30)

# Convert using half()/float()/double()/int()/long()/short() methods
tensor_half = tensor_float.half()
print(f"Tensor converted to half (float16):\n{tensor_half}")
print(f"elements type: {tensor_half.dtype}, type: {type(tensor_half)}")
print('-' * 30)

tensor_double = tensor_float.double()
print(f"Tensor converted to double (float64):\n{tensor_double}")
print(f"elements type: {tensor_double.dtype}, type: {type(tensor_double)}")
print('-' * 30)

tensor_int = tensor_float.int()
print(f"Tensor converted to int (int32):\n{tensor_int}")
print(f"elements type: {tensor_int.dtype}, type: {type(tensor_int)}")
print('-' * 30)

tensor_long = tensor_float.long()
print(f"Tensor converted to long (int64):\n{tensor_long}")
print(f"elements type: {tensor_long.dtype}, type: {type(tensor_long)}")
print('-' * 30)

tensor_short = tensor_float.short()
print(f"Tensor converted to short (int16):\n{tensor_short}")
print(f"elements type: {tensor_short.dtype}, type: {type(tensor_short)}")
print('-' * 30)

tensor_float_again = tensor_int.float()
print(f"Tensor converted back to float (float32):\n{tensor_float_again}")
print(f"elements type: {tensor_float_again.dtype}, type: {type(tensor_float_again)}")
print('-' * 30)
