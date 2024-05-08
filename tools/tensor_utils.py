'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import torch
import random
from torch.nn.utils.rnn import pad_sequence

def adjust_tensor_dim(tensor, target_dim = 3):
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

def generate_padding_mask(source_batch):
    """
    Generate a padding mask for the source batch where all -inf values are masked.
    Args:
    source_batch (torch.Tensor): Tensor of shape [batch_size, seq_len, obs_size]
    Returns:
    torch.Tensor: A 3D tensor of shape [batch_size, seq_len, 1] where padded elements are 1.
    """
    if source_batch.dim() != 3:
        return None
    
    # Identify positions that are -inf (assuming -inf represents padding)
    padding_mask = (source_batch == float('-inf')).any(dim=-1)

    padding_mask = ~padding_mask
    
    return padding_mask.unsqueeze(-1).float()

def get_random_batch(eval_dataset, batch_size):
    num_batches = len(eval_dataset) // batch_size
    random_index = random.randint(0, num_batches - 1) if num_batches > 0 else 0
    start_index = random_index * batch_size
    end_index = start_index + batch_size

    batch = [eval_dataset[i] for i in range(start_index, min(end_index, len(eval_dataset)))]
    source_batch, target_batch = zip(*batch)

    # Check if the elements are tensors, convert only if they are not
    source_batch = pad_sequence([s if isinstance(s, torch.Tensor) else torch.tensor(s) for s in source_batch],
                                batch_first=True, padding_value=0)
    target_batch = pad_sequence([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in target_batch],
                                batch_first=True, padding_value=0)

    return source_batch, target_batch
    
def convert_to_device(source_batch, target_batch, device):
    source_batch, target_batch = source_batch.float().to(device), target_batch.float().to(device)
    return source_batch, target_batch

def encode_inputs(encoder, observation, labels):
    with torch.no_grad():
        encoded_obseartion = observation if encoder is None else encoder.encode(observation)
    return encoded_obseartion, labels