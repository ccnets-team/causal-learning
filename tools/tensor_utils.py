'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import torch
import random
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

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

def majority_voting(predictions, weights=None):
    """
    Applies majority voting to determine the most common class across sequences, considering weights.

    Parameters:
    - predictions (Tensor): The input tensor of shape [S].
    - weights (Tensor): The weights for each prediction of shape [S].

    Returns:
    - Tensor: The most frequent class index.
    """
    if len(predictions) == 0:
        return torch.tensor(-1)  # Return a placeholder value if no valid predictions
    
    if weights is None:
        weights = torch.ones_like(predictions, dtype=torch.float)
    
    weighted_counts = Counter()
    for pred, weight in zip(predictions.tolist(), weights.tolist()):
        weighted_counts[pred] += weight
    
    majority_vote = max(weighted_counts, key=weighted_counts.get)

    return torch.tensor(majority_vote)  # Return as tensor

def aggregate_test_elements(valid_seqs, task_type, num_classes=None):
    """
    Aggregates predictions over the sequence according to the task type.

    Parameters:
    - valid_seqs (list of Tensors): List of valid tensors for each sequence with shape [S, F].
    - task_type (str): The type of task. Can be 'binary_classification', 'multi_class_classification', 'multi_label_classification', 'regression'.
    - num_classes (int): The number of classes for classification tasks.

    Returns:
    - aggregated_list (Tensor): A tensor containing the aggregated elements.
    """
    aggregated_list = []

    for valid_seq in valid_seqs:
        seq_length = valid_seq.size(0)
        weights = torch.arange(1, seq_length + 1, dtype=torch.float)
        
        if task_type == 'binary_classification':
            binary_predictions = (valid_seq > 0.5).int()
            vote = majority_voting(binary_predictions, weights)
            aggregated_list.append(vote)
        elif task_type == 'multi_class_classification':
            class_indices = torch.argmax(valid_seq, dim=-1)
            vote = majority_voting(class_indices, weights)
            one_hot_vote = torch.nn.functional.one_hot(vote, num_classes=num_classes).float()
            aggregated_list.append(one_hot_vote)
        elif task_type == 'multi_label_classification':
            votes = torch.stack([majority_voting(valid_seq[:, j], weights) for j in range(num_classes)])
            aggregated_list.append(votes)
        elif task_type == 'regression':
            avg_value = (valid_seq * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
            aggregated_list.append(avg_value)

    aggregated_list = torch.stack(aggregated_list).to(valid_seqs[0].device)
    return aggregated_list

def select_elements_for_testing(src, dst, padding_mask, task_type, num_classes=None):
    """
    Extracts valid elements of src and dst sequences using the padding mask and applies appropriate aggregation.

    Parameters:
    - src (Tensor): The source batch tensor of shape [B, S, F].
    - dst (Tensor): The destination batch tensor of shape [B, S, num_classes].
    - padding_mask (Tensor): The padding mask tensor of shape [B, S, 1] where padding is marked by 1s.
    - task_type (str): The type of task. Can be 'binary_classification', 'multi_class_classification', 'multi_label_classification', 'regression'.
    - num_classes (int): The number of classes for classification tasks.

    Returns:
    - selected_src (Tensor): A tensor containing the aggregated elements of src with shape [B, F].
    - selected_dst (Tensor): A tensor containing the aggregated elements of dst with shape [B, num_classes].
    """
    padding_mask = padding_mask.squeeze(dim=-1)  # Remove the last dimension
    
    valid_src = [src[i][padding_mask[i].bool()] for i in range(src.size(0))]
    valid_dst = [dst[i][padding_mask[i].bool()] for i in range(dst.size(0))]

    # Stack the valid elements back into tensors
    valid_src = torch.nn.utils.rnn.pad_sequence(valid_src, batch_first=True)
    valid_dst = torch.nn.utils.rnn.pad_sequence(valid_dst, batch_first=True)

    # Aggregate src and dst according to the task type
    aggregated_src = aggregate_test_elements(valid_src, task_type, num_classes)
    aggregated_dst = aggregate_test_elements(valid_dst, task_type, num_classes)
    
    aggregated_src = adjust_tensor_dim(aggregated_src, target_dim=3)
    aggregated_dst = adjust_tensor_dim(aggregated_dst, target_dim=3)

    return aggregated_src, aggregated_dst

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