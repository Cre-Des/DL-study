'''
An example of tensor auto-differentiation using PyTorch.

Important functions:
- torch.autograd.grad: Computes and returns the sum of gradients of outputs with respect to the inputs.
- torch.autograd.backward: Computes the sum of gradients of outputs with respect to the graph leaves.

- An example of tensor auto-differentiation is provided in the function `_example_tensors_auto_diff`.
- A gradient descent example is provided in the function `gradient_descent_example`.

'''

# Import necessary libraries
import torch

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


if __name__ == '__main__':
    #_example_tensors_auto_diff()
    gradient_descent_example()