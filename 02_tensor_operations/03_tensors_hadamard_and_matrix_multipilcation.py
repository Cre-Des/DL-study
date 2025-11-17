'''
A example to illustrate tensor hadamard product and matrix multiplication operations.

Important operations:
    - Hadamard Product (Element-wise Multiplication): 
            tensor1 * tensor2
            torch.mul(tensor1, tensor2) 
            tips: two tensors must have the same shape.
    - Matrix Multiplication: 
            tensor1 @ tensor2 
            torch.matmul(tensor1, tensor2)
            tensor1.dot(tensor2) tips: for 1-D tensors
            tips: follow the matrix multiplication rules.

    -Matrix multiplication rules:
        - If both tensors are 1-D, the dot product (scalar) is returned.
        - If both arguments are 2-D, the matrix-matrix product is returned.
        - If the first argument is 1-D and the second argument is 2-D, a 1-D tensor is returned.
        - If the first argument is 2-D and the second argument is 1-D, a 1-D tensor is returned.
        - If both arguments are at least 1-D and at least one argument is N-D (where N > 2), it is treated as a stack of matrices residing in the last two dimensions and broadcast accordingly.
'''

# Import necessary libraries
import torch

# 1. Hadamard Product (Element-wise Multiplication)
def hadamard_product():
    # Create sample tensors
    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
    print(f"Tensor1:\n{tensor1}")
    print(f"Tensor2:\n{tensor2}")

    # Hadamard Product using * operator
    hadamard_result1 = tensor1 * tensor2
    print(f"Hadamard Product using * operator:\n{hadamard_result1}")

    # Hadamard Product using torch.mul()
    hadamard_result2 = torch.mul(tensor1, tensor2)
    print(f"Hadamard Product using torch.mul():\n{hadamard_result2}")
    print("=" * 40)

# 2. Matrix Multiplication
def matrix_multiplication():
    # Create sample tensors
    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]]) # Shape (2, 3)
    tensor2 = torch.tensor([[7, 8], [9, 10], [11, 12]]) # Shape (3, 2)
    print(f"Tensor1:\n{tensor1}")
    print(f"Tensor2:\n{tensor2}")

    # Matrix Multiplication using @ operator
    matmul_result1 = tensor1 @ tensor2
    print(f"Matrix Multiplication using @ operator:\n{matmul_result1}")

    # Matrix Multiplication using torch.matmul()
    matmul_result2 = torch.matmul(tensor1, tensor2)
    print(f"Matrix Multiplication using torch.matmul():\n{matmul_result2}")

    # Matrix Multiplication using dot() for 1-D tensors
    tensor3 = torch.tensor([1, 2, 3])
    tensor4 = torch.tensor([4, 5, 6])
    dot_result = tensor3.dot(tensor4)
    print(f"Dot Product of 1-D tensors using dot(): {dot_result}")
    print("=" * 40)

# Test functions
if __name__ == "__main__":
    hadamard_product()
    matrix_multiplication()