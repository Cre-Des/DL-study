'''
A example to illustrate tensor basic computation operations.

Important operations:
    - Addition: tensor1 + tensor2 or torch.add(tensor1, tensor2)
    - Subtraction: tensor1 - tensor2 or torch.sub(tensor1, tensor2)
    - Multiplication: tensor1 * tensor2 or torch.mul(tensor1, tensor2)
    - Division: tensor1 / tensor2 or torch.div(tensor1, tensor2)
    - Negation: -tensor or torch.neg(tensor)

    Inplace operations:
    - add_()
    - sub_()
    - mul_()
    - div_()
    - neg_()

If operations between tensors and numbers, the number will be operated with each element of the tensor.
'''

# Import necessary libraries
import torch

# 1. Basic Tensor Computation
# Create sample tensors
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
number = 10
print(f"Tensor1: {tensor1}")
print(f"Tensor2: {tensor2}")
print(f"Number: {number}")

# Addition
add_result = tensor1 + tensor2
print(f"Addition (tensor1 + tensor2): {add_result}")
add_result_num = tensor1 + number
print(f"Addition (tensor1 + number): {add_result_num}")
add_result_num2 = tensor1.add(number)
print(f"Addition using torch.add (tensor1 + number): {add_result_num2}")

tensor_inplace = torch.tensor([1, 2, 3])
print(f"\nOriginal Tensor for Inplace Operations: {tensor_inplace}")
tensor_inplace.add_(5)
print(f"Inplace Addition (add_ 5): {tensor_inplace}")

# Another way of inplace addition
tensor_inplace = torch.tensor([1, 2, 3])
tensor_inplace += 5
print(f"Inplace Addition (+= 5): {tensor_inplace}")
print("=" * 40)

# Subtraction
sub_result = tensor2 - tensor1
print(f"Subtraction (tensor2 - tensor1): {sub_result}")
sub_result_num = tensor2 - number
print(f"Subtraction (tensor2 - number): {sub_result_num}")
sub_result_num2 = tensor2.sub(number)
print(f"Subtraction using torch.sub (tensor2 - number): {sub_result_num2}")

tensor_inplace = torch.tensor([4, 5, 6])
print(f"\nOriginal Tensor for Inplace Operations: {tensor_inplace}")
tensor_inplace.sub_(2)
print(f"Inplace Subtraction (sub_ 2): {tensor_inplace}")

# Another way of inplace subtraction
tensor_inplace = torch.tensor([4, 5, 6])
tensor_inplace -= 2
print(f"Inplace Subtraction (-= 2): {tensor_inplace}")
print("=" * 40)

# Multiplication
mul_result = tensor1 * tensor2
print(f"Multiplication (tensor1 * tensor2): {mul_result}")
mul_result_num = tensor1 * number
print(f"Multiplication (tensor1 * number): {mul_result_num}")
mul_result_num2 = tensor1.mul(number)
print(f"Multiplication using torch.mul (tensor1 * number): {mul_result_num2}")

tensor_inplace = torch.tensor([1, 2, 3])
print(f"\nOriginal Tensor for Inplace Operations: {tensor_inplace}")
tensor_inplace.mul_(3)
print(f"Inplace Multiplication (mul_ 3): {tensor_inplace}")

# Another way of inplace multiplication
tensor_inplace = torch.tensor([1, 2, 3])
tensor_inplace *= 3
print(f"Inplace Multiplication (*= 3): {tensor_inplace}")
print("=" * 40)

# Division
div_result = tensor2 / tensor1
print(f"Division (tensor2 / tensor1): {div_result}")
div_result_num = tensor2 / number
print(f"Division (tensor2 / number): {div_result_num}")
div_result_num2 = tensor2.div(number)
print(f"Division using torch.div (tensor2 / number): {div_result_num2}")

tensor_inplace = torch.tensor([4.0, 5.0, 6.0])
print(f"\nOriginal Tensor for Inplace Operations: {tensor_inplace}")
tensor_inplace.div_(2)
print(f"Inplace Division (div_ 2): {tensor_inplace}")

# Another way of inplace division
tensor_inplace = torch.tensor([4.0, 5.0, 6.0])
tensor_inplace /= 2
print(f"Inplace Division (/= 2): {tensor_inplace}")
print("=" * 40)

# Negation
neg_result = -tensor1
print(f"Negation (-tensor1): {neg_result}")
neg_result2 = torch.neg(tensor1)
print(f"Negation using torch.neg (tensor1): {neg_result2}")

tensor_inplace = torch.tensor([1, -2, 3])
print(f"\nOriginal Tensor for Inplace Operations: {tensor_inplace}")
tensor_inplace.neg_()
print(f"Inplace Negation (neg_): {tensor_inplace}")
print("=" * 40)
