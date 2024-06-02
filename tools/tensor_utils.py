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

def manage_batch_dimensions(use_seq, source_batch, target_batch, padding_mask = None):
    """
    Prepares batch data by adjusting tensor dimensions for processing.

    Parameters:
    - source_batch (Tensor): The source batch tensor.
    - target_batch (Tensor): The target batch tensor.
    - padding_mask (Tensor, optional): The padding mask tensor.

    Returns:
    - state_trajectory (Tensor): The prepared source batch tensor.
    - target_trajectory (Tensor): The prepared target batch tensor.
    - padding_mask (Tensor, optional): The prepared padding mask tensor.
    """
            
    if use_seq:
        # Adjust tensor dimensions for causal processing
        state_trajectory = adjust_tensor_dim(source_batch, target_dim=3)  # off when it's img data set
        target_trajectory = adjust_tensor_dim(target_batch, target_dim=3)  # off when it's img data set
        padding_mask = adjust_tensor_dim(padding_mask, target_dim=3)  # off when it's img data set
    else:
        state_trajectory = source_batch
        target_trajectory = target_batch
    
    return state_trajectory, target_trajectory, padding_mask

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

def prepare_batches(source_batch, target_batch, label_size, task_type, device):
    """
    Prepares batches by moving them to the appropriate device, converting target batch to one-hot encoding,
    and generating a padding mask.

    Parameters:
    - source_batch (Tensor): The source batch tensor.
    - target_batch (Tensor): The target batch tensor.
    - label_size (int): The size of the labels for one-hot encoding.
    - task_type (str): The type of task for one-hot encoding.
    - device (torch.device): The device to move the batches to.

    Returns:
    - source_batch (Tensor): The source batch tensor on the appropriate device.
    - target_batch (Tensor): The one-hot encoded target batch tensor on the appropriate device.
    - padding_mask (Tensor): The generated padding mask tensor.
    """
    # Prepare batches by moving them to the appropriate device.
    source_batch, target_batch = convert_to_device(source_batch, target_batch, device=device)
    target_batch = convert_to_one_hot(target_batch, label_size=label_size, task_type=task_type)
    source_batch, target_batch, padding_mask = generate_padding_mask(source_batch, target_batch)

    return source_batch, target_batch, padding_mask

def select_last_sequence_elements(src, dst, padding_mask):
    """
    Extracts the last elements of src and dst sequences using the padding mask.

    Parameters:
    - src (Tensor): The source batch tensor of shape [B, S, F].
    - dst (Tensor): The destination batch tensor of shape [B, S, L].
    - padding_mask (Tensor): The padding mask tensor of shape [B, S, 1] where padding is marked by 1s.

    Returns:
    - selected_src (Tensor): A tensor containing only the last non-padding elements of src with shape [B, 1, F].
    - selected_dst (Tensor): A tensor containing only the last non-padding elements of dst with shape [B, 1, L].
    """
    padding_mask = padding_mask.squeeze(dim=-1)  # Remove the last dimension
    
    # Calculate the cumulative sum along the sequence dimension (reverse order)
    last_indices = padding_mask.size(1) - 1 - padding_mask.flip(dims=[1]).argmax(dim=1)
    not_fully_padded = padding_mask.any(dim=1)

    # Gather the last elements using the indices
    batch_size = src.size(0)
    
    selected_src = src[torch.arange(batch_size, device=src.device), last_indices, :].unsqueeze(1)
    selected_dst = dst[torch.arange(batch_size, device=src.device), last_indices, :].unsqueeze(1)
    
    selected_src = selected_src[not_fully_padded]
    selected_dst = selected_dst[not_fully_padded]
    return selected_src, selected_dst

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

def convert_to_one_hot(target_batch, label_size, task_type):
    if target_batch is None or label_size is None or task_type is None:
        return None
    
    # Mask to identify locations with -1 which should remain unchanged after encoding
    target_label_size = None
    if task_type == 'binary_classification':
        target_label_size = 2
    elif task_type == 'multi_class_classification':
        target_label_size = label_size

    if target_label_size is not None and target_batch.shape[-1] != target_label_size:
        mask = target_batch == -1
        target_batch[mask] = 0
        target_batch = torch.nn.functional.one_hot(target_batch.long(), num_classes=target_label_size).float().squeeze(-2)
        expanded_mask = mask.expand_as(target_batch)    
        target_batch[expanded_mask] = -1
    return target_batch
