'''
An example to create linear and random tensors.

functions.
    - torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    - torch.linspace(start, end, steps=100, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

    - torch.random.initial_seed() → int
    - torch.random.manual_seed(seed) → torch.Generator
    - torch.rand(size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    - torch.randn(size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    - torch.randint(low, high, size, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) → Tensor

'''

# Import necessary libraries
import torch

# 1. Create linear tensors using torch.arange() and torch.linspace()
def create_linear_tensors():
    # 1.1 Create a tensor with values from 0 to 9 using torch.arange()
    '''
        torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
            - start: starting value of the sequence (inclusive).
            - end: ending value of the sequence (exclusive).
            - step: difference between each pair of consecutive values.
    '''
    tensor_arange = torch.arange(0, 10, 1)
    print(f"Tensor using arange (0 to 9):\n{tensor_arange}")
    print(f"type: {type(tensor_arange)}")
    print('-' * 30)

    # 1.2 Create a tensor with 5 evenly spaced values from 0 to 1 using torch.linspace()
    '''
        torch.linspace(start, end, steps=100, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
            - start: starting value of the sequence.
            - end: ending value of the sequence.
            - steps: number of values to generate between start and end (inclusive).
    '''
    tensor_linspace = torch.linspace(0, 1, steps=5)
    print(f"Tensor using linspace (0 to 1 with 5 steps):\n{tensor_linspace}")
    print(f"type: {type(tensor_linspace)}")
    print('-' * 30)

# 2. Create random tensors using torch.rand(), torch.randn() and torch.randint()
def create_random_tensors():

    # Step to set random seed for reproducibility 
    # torch.initial_seed() # Get the current random seed
    torch.manual_seed(42) # Set the random seed to a fixed value
    '''
        - torch.manual_seed(seed) → torch.Generator
            - seed: the desired seed value (integer).
        for all subsequent random tensor generations.
        we can write torch.manual_seed(torch.initial_seed()) after import necessary libraries
    '''
    
    # 2.1 Create a 2x3 tensor with random values from a uniform distribution [0, 1) using torch.rand()
    tensor_rand = torch.rand(2, 3)
    print(f"Tensor with random values (uniform distribution):\n{tensor_rand}")
    print(f"type: {type(tensor_rand)}")
    print('-' * 30)

    # 2.2 Create a 2x3 tensor with random values from a normal distribution using torch.randn()
    tensor_randn = torch.randn(2, 3)
    print(f"Tensor with random values (normal distribution):\n{tensor_randn}")
    print(f"type: {type(tensor_randn)}")
    print('-' * 30)

    # 2.3 Create a 2x3 tensor with random integer values between 0 and 10 using torch.randint()
    tensor_randint = torch.randint(0, 10, (2, 3))
    print(f"Tensor with random integer values (0 to 9):\n{tensor_randint}")
    print(f"type: {type(tensor_randint)}")
    print('-' * 30)


# 3. Test the functions
if __name__ == "__main__":
    print("Creating Linear Tensors:")
    create_linear_tensors()
    
    print("Creating Random Tensors:")
    create_random_tensors()
