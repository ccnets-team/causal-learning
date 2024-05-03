'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import torch

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

def convert_to_device(source_batch, target_batch, device):
    source_batch, target_batch = source_batch.float().to(device), target_batch.float().to(device)
    return source_batch, target_batch

def generate_padding_mask(source_batch):
    """
    Generate a padding mask for the source batch where all -inf values are masked.
    Args:
    source_batch (torch.Tensor): Tensor of shape [batch_size, seq_len, obs_size]
    Returns:
    torch.Tensor: A 3D tensor of shape [batch_size, seq_len, 1] where padded elements are 1.
    """
    # Identify positions that are -inf (assuming -inf represents padding)
    padding_mask = (source_batch == float('-inf')).any(dim=-1)

    padding_mask = ~padding_mask
    
    return padding_mask.unsqueeze(-1).float()

def encode_inputs(encoder, observation, labels):
    with torch.no_grad():
        encoded_obseartion = observation if encoder is None else encoder.encode(observation)
    return encoded_obseartion, labels
