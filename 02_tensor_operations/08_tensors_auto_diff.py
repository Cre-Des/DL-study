'''
An example of tensor auto-differentiation using PyTorch.

Important functions:
- torch.grad: Computes and returns the sum of gradients of outputs with respect to the inputs.
- torch.backward: Computes the sum of gradients of outputs with respect to the graph leaves.
- detach(): if a tensor is set auto-differentiation, it can't be convert to numpy array.

- An example of tensor auto-differentiation is provided in the function `_example_tensors_auto_diff`.
- A gradient descent example is provided in the function `gradient_descent_example`.
- An example of detach()

'''

# Import necessary libraries
import torch
import numpy as np

def _example_tensors_auto_diff():
    # 1. define initial w tensor with requires_grad=True to track computation
    w = torch.tensor(10, requires_grad=True, dtype=torch.float)

    # 2. define loss function
    loss = 2*w**2 # loss = 2 * (w^2)

    print(f'Initial w: {w.item()}')
    print(f'Initial loss: {type(loss.grad_fn)}') # Check the type of grad_fn

    # 4. compute gradients using torch.autograd.grad
    # loss.sum() is used to convert loss to a scalar value
    # loss.sum().backward() to ensure compatibility with autograd
    loss.backward()

    # 5. update w using gradient descent
    assert w.grad is not None
    w.data = w.data - 0.01 * w.grad

    print(f'Updated w: {w.item()}')
    print(f'Gradient of w: {w.grad}')

# Gradient descent example
def gradient_descent_example():
    # Initialize w
    w = torch.tensor(10, requires_grad=True, dtype=torch.float)

    # Define loss function
    loss = w**2 + 20   # loss = w^2 + 20
    learning_rate = 0.01

    # Perform gradient descent for 100 iterations
    print('Starting gradient descent...')
    print(f'Initial w: {w},(0.01 * w.grad): None, loss:{loss}')
    # forward and backward in each iteration
    for step in range(100):
       
        # forward pass: compute loss
        loss = w**2 + 20  # Recompute loss in each iteration

        # Zero gradients before backward pass (important!  to avoid accumulation)
        if w.grad is not None:
            w.grad.zero_()

        # Compute gradients
        loss.sum().backward()
        assert w.grad is not None

        # Update w using gradient descent
        w.data = w.data - learning_rate * w.grad

        print(f'Step {step+1}: w: {w.item()}, (0.01 * w.grad): {learning_rate * w.grad}, loss:{loss}')    

    print(f'Final w after gradient descent: {w.item()}')
    print(f'Final loss after gradient descent: {loss.item()}') 

def use_detech():
    tensor = torch.tensor([10, 20], dtype=torch.float)
    print(f'Original tensor: {tensor}, type:{type(tensor)}')

    tensor_diff = torch.tensor([10, 20], requires_grad= True, dtype=torch.float)
    print(f'Original tensor: {tensor_diff}, type:{type(tensor_diff)}')

    numpy_array = tensor.numpy()
    print(f'Original tensor to numpy: {numpy_array}, type:{type(numpy_array)}')

    # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    # numpy_array_diff = tensor_diff.numpy()
    # print(f'Original tensor: {numpy_array_diff}, type:{type(numpy_array_diff)}')

    tensor_diff_detach = tensor_diff.detach()
    print(f'Detach tensor: {tensor_diff_detach}, type:{type(tensor_diff_detach)}')
    print('-' * 30)

    # Share memory
    tensor_diff.data[0] = 100
    print(f'Original tensor: {tensor_diff}, type:{type(tensor_diff)}')
    print(f'Detach tensor: {tensor_diff_detach}, type:{type(tensor_diff_detach)}')
    print('-' * 30)

    # Show who can auto-diff
    print(f'tensor_diff:{tensor_diff.requires_grad}')
    print(f'tensor_diff_detach:{tensor_diff_detach.requires_grad}')
    print('-' * 30)

    numpy_diff_detach = tensor_diff_detach.numpy()
    print(f'Original tensor to numpy: {numpy_diff_detach}, type:{type(numpy_diff_detach)}')
    print('-' * 30)

    # Final
    numpy_diff_detach = tensor_diff.detach().numpy()
    print(f'Original tensor to numpy: {numpy_diff_detach}, type:{type(numpy_diff_detach)}')



if __name__ == '__main__':
    #_example_tensors_auto_diff()
    #gradient_descent_example()
    use_detech()