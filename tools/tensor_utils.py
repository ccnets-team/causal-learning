'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import torch
import random
from torch.nn.utils.rnn import pad_sequence

def adjust_tensor_dim(tensor, target_dim = 3):
    if tensor is None:
        return None
    
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
                tensor = tensor.squeeze(0)  # Squeeze dimension at index 1 if it's 1
    return tensor

def generate_padding_mask(source_data, target_data, padding_values = [0, -1]):
    """
    Generate a padding mask for the source data where padding values are masked and apply it.
    Args:
    source_data (torch.Tensor): Tensor of shape [batch_size, seq_len, obs_size]
    Returns:
    torch.Tensor: The source_data tensor with padded positions zeroed out.
    torch.Tensor: A boolean tensor of shape [batch_size, seq_len, 1] indicating padding.
    """
    if source_data.dim() != 3:
        return source_data, target_data, None
    
    # Identify padding positions
    padding_positions_x = (source_data == padding_values[0]).any(dim=-1)
    padding_positions_y = (target_data == padding_values[1]).any(dim=-1) if target_data is not None else torch.ones_like(padding_positions_x)
    padding_positions = padding_positions_x & padding_positions_y
    # Create a mask where true values indicate non-padding, false indicate padding
    padding_mask = ~padding_positions
    padding_mask = padding_mask.unsqueeze(-1).float()
    
    # Expand the mask to the size of source_data for element-wise multiplication
    expanded_non_padding_mask = padding_mask.expand_as(source_data)

    # Zero out padding positions in the source_data and target_data
    source_data = source_data * expanded_non_padding_mask
    if target_data is not None:
        target_data = target_data * padding_mask

    return source_data, target_data, padding_mask

def get_random_batch(dataset, batch_size):
    num_batches = len(dataset) // batch_size
    random_index = random.randint(0, num_batches - 1) if num_batches > 0 else 0
    start_index = random_index * batch_size
    end_index = start_index + batch_size

    batch = [dataset[i] for i in range(start_index, min(end_index, len(dataset)))]
    source_batch, target_batch = zip(*batch)

    # Convert elements to tensors only if they are not already tensors
    source_batch = pad_sequence(source_batch, batch_first=True, padding_value=0)
    target_batch = pad_sequence(target_batch, batch_first=True, padding_value=-1)

    return source_batch, target_batch

def convert_to_device(source_batch, target_batch, device):
    source_batch = source_batch.float().to(device)
    if target_batch is not None:
        target_batch = target_batch.float().to(device)
    return source_batch, target_batch

def encode_inputs(encoder, observation, labels):
    with torch.no_grad():
        encoded_obseartion = observation if encoder is None else encoder.encode(observation)
    return encoded_obseartion, labels