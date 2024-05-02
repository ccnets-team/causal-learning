'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''

def adjust_tensor_dim(tensor, target_dim = 3):
    # Ensure the tensor has at least two dimensions to avoid index errors
    while tensor.dim() < 2:
        tensor = tensor.unsqueeze(0)  # Add a dimension at the front if less than 2 dims

    # Determine how many dimensions need to be added or removed
    required_dims = target_dim - tensor.dim()
    if required_dims > 0:
        for _ in range(required_dims):
            tensor = tensor.unsqueeze(1)  # Add new dimensions at index 1
    elif required_dims < 0:
        for _ in range(-required_dims):
            if tensor.size(1) == 1:
                tensor = tensor.squeeze(1)  # Squeeze dimension at index 1 if it's 1
            else:
                break  # Do not squeeze if the dimension size is not 1
    return tensor