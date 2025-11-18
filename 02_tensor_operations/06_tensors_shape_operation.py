'''
An example of tensor shape operations using PyTorch.

Classifications:
        reshape() - reshapes a tensor to a specified shape without changing its data.
        squeeze() - removes dimensions of size 1 from the tensor.
        unsqueeze() - adds a dimension of size 1 at the specified position.
        transpose() - swaps two specified dimensions of a tensor.
        permute() - rearranges all dimensions of a tensor according to a specified order.
        view() - returns a new tensor with the same data but different shape.(must be contiguous in memory)
        contiguous() - returns a contiguous tensor containing the same data.
        is_contiguous() - checks if the tensor is contiguous in memory.
'''

# Import necessary libraries
import torch

# set random seed for reproducibility
torch.manual_seed(24)

# 1. reshape()
# Reshapes a tensor to a specified shape without changing its data.
# The total number of elements must remain the same.
def reshape_tensor():
        # Create a 2x3 tensor with random values
        tensor = torch.randint(1,10,(2, 3))
        print("Original Tensor:\n", tensor)
        print(f'Original Shape: {tensor.shape},rows: {tensor.shape[0]}, columns: {tensor.shape[1]}')
        print('-'*50)
        
        # Reshape to 3x2
        reshaped_tensor = tensor.reshape(3, 2)
        print("Reshaped Tensor (3x2):\n", reshaped_tensor)
        print(f'Reshaped Shape: {reshaped_tensor.shape},rows: {reshaped_tensor.shape[0]}, columns: {reshaped_tensor.shape[1]}')
        print('-'*50)

        # Reshape to 1x6
        reshaped_tensor_1x6 = tensor.reshape(1, 6)
        print("Reshaped Tensor (1x6):\n", reshaped_tensor_1x6)
        print(f'Reshaped Shape: {reshaped_tensor_1x6.shape},rows: {reshaped_tensor_1x6.shape[0]}, columns: {reshaped_tensor_1x6.shape[1]}')
        print('='*50)

        # Reshape to 2x5
        # reshaped_tensor_2x5 = tensor.reshape(2, 5)
        # print("Reshaped Tensor (2x5):\n", reshaped_tensor_2x5)
        # print(f'Reshaped Shape: {reshaped_tensor_2x5.shape},rows: {reshaped_tensor_2x5.shape[0]}, columns: {reshaped_tensor_2x5.shape[1]}')     
        # print('='*50)

        # shape '[2, 5]' is invalid for input of size 6
        # The above line will raise an error because the total number of elements must remain the same during reshaping.

# 2. squeeze() and unsqueeze()
# squeeze() - removes dimensions of size 1 from the tensor.
# unsqueeze() - adds a dimension of size 1 at the specified position.
def squeeze_unsqueeze_tensor():
        # Create a tensor with shape (1, 3, 1, 4)
        tensor = torch.randint(1, 10, (1, 3, 1, 4))
        print("Original Tensor:\n", tensor)
        print("Original Tensor Shape:", tensor.shape)
        print('-'*50)

        # Squeeze the tensor
        squeezed_tensor = tensor.squeeze()
        print("Squeezed Tensor:\n", squeezed_tensor)
        print("Squeezed Tensor Shape:", squeezed_tensor.shape)
        print('-'*50)

        # Unsqueeze the tensor at dimension 0
        unsqueezed_tensor = squeezed_tensor.unsqueeze(0)
        print("Unsqueezed Tensor:\n", unsqueezed_tensor)
        print("Unsqueezed Tensor Shape (dim=0):", unsqueezed_tensor.shape)
        print('-'*50)

        # Unsqueeze the tensor at dimension 2
        unsqueezed_tensor_dim2 = squeezed_tensor.unsqueeze(2)
        print("Unsqueezed Tensor (dim=2):\n", unsqueezed_tensor_dim2)
        print("Unsqueezed Tensor Shape (dim=2):", unsqueezed_tensor_dim2.shape)
        print('-'*50)

# 3. transpose() and permute()
# transpose() - swaps two specified dimensions of a tensor.
# permute() - rearranges all dimensions of a tensor according to a specified order.
def transpose_tensor():
        # Create a 2x3 tensor
        tensor = torch.randint(1, 10, (2, 3, 4))
        print("Original Tensor:\n", tensor)
        print("Original Tensor Shape:", tensor.shape)
        print('-'*50)

        # Transpose the tensor)
        transposed_tensor = tensor.transpose(0, 1)
        print("Transposed Tensor:\n", transposed_tensor)
        print("Original Tensor:\n", tensor)
        print("Transposed Tensor Shape:", transposed_tensor.shape)
        print('-'*50)

        # Permute the tensor(rearranging all dimensions)
        permuted_tensor = tensor.permute(2, 0, 1)
        print("Permuted Tensor:\n", permuted_tensor)
        print("Permuted Tensor Shape:", permuted_tensor.shape)
        print('='*50)

# 4. view(), contiguous(), is_contiguous()
# view() - returns a new tensor with the same data but different shape.(must be contiguous in memory)
# contiguous() - returns a contiguous tensor containing the same data.
# is_contiguous() - checks if the tensor is contiguous in memory.
def view_contiguous_iscontiguous():
        # Create a 2x3 tensor
        tensor = torch.randint(1, 10, (2, 3, 4))
        print("Original Tensor:\n", tensor)
        print("Original Tensor Shape:", tensor.shape)
        print("Is Original Tensor Contiguous?:", tensor.is_contiguous())
        print('-'*50)

        # View the tensor as 6x2
        viewed_tensor = tensor.view(6, 2, 2)
        print("Viewed Tensor (6x2x2):\n", viewed_tensor)
        print("Viewed Tensor Shape:", viewed_tensor.shape)
        print("Is Viewed Tensor Contiguous?:", viewed_tensor.is_contiguous())
        print('-'*50)

        # Transpose the tensor to make it non-contiguous
        transposed_tensor = tensor.transpose(0, 1)
        print("Transposed Tensor:\n", transposed_tensor)
        print("Transposed Tensor Shape:", transposed_tensor.shape)
        print("Is Transposed Tensor Contiguous?:", transposed_tensor.is_contiguous())
        print('-'*50)

        # View the transposed tensor (this will raise an error if uncommented)
        # viewed_transposed_tensor = transposed_tensor.view(6, 2, 2)
        # print("Viewed Transposed Tensor (6x2x2):\n", viewed_transposed_tensor)
        # print("Viewed Transposed Tensor Shape:", viewed_transposed_tensor.shape)
        # print("Is Viewed Transposed Tensor Contiguous?:", viewed_transposed_tensor.is_contiguous())
        # print('-'*50)

        # Make the transposed tensor contiguous
        contiguous_tensor = transposed_tensor.contiguous()
        print("Contiguous Tensor:\n", contiguous_tensor)
        print("Contiguous Tensor Shape:", contiguous_tensor.shape)
        print("Is Contiguous Tensor Contiguous?:", contiguous_tensor.is_contiguous())
        print('-'*50)


# Test function
if __name__ == "__main__":
        reshape_tensor()
        squeeze_unsqueeze_tensor()
        transpose_tensor()
        view_contiguous_iscontiguous()