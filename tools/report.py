'''
Author:
        
        PARK, JunHo, junho@ccnets.org
        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def transform_labels_to_original_scale(inferred_y, target_y, task_type, label_size, label_scale):
    if label_scale is not None and 'regression' in task_type:
        if len(label_scale) == label_size:
            tensor_scale = torch.tensor(label_scale).to(inferred_y.device)
            inferred_y = inferred_y * tensor_scale
            target_y = target_y * tensor_scale
    return inferred_y, target_y

def calculate_test_results(inferred_y, target_y, task_type, label_size, label_scale):
    inferred_y, target_y = transform_labels_to_original_scale(inferred_y, target_y, task_type, label_size, label_scale)
        
    if task_type == 'binary_classification':
        inferred_y = (inferred_y > 0.5).int()
        target_y = (target_y > 0.5).int()
    elif task_type == 'multi_class_classification':
        inferred_y = torch.argmax(inferred_y, dim=-1)
        target_y = torch.argmax(target_y, dim=-1)  # Assuming one-hot encoding of target
    elif task_type == "ordinal_regression":
        inferred_y = torch.round(inferred_y).int().clamp(0, label_size - 1)
        target_y = torch.round(target_y).int()
    
    if task_type == 'binary_classification':
        num_classes = 2
    else:
        num_classes = label_size
        
    return get_test_results(inferred_y, target_y, task_type, num_classes)

def convert_to_tensor(values):
    """
    Converts input values into a PyTorch tensor with appropriate data type handling.
    Supports numpy arrays, lists, and other iterable data types, with specific handling based on the data type of the first element in lists.
    
    Parameters:
    - values: The input data to convert. Can be a numpy array, list, or other iterable types.
    
    Returns:
    - A PyTorch tensor converted from the input data.
    """
    # First, handle cases where values are already a torch.Tensor
    if isinstance(values, torch.Tensor):
        return values

    # For numpy arrays, handle different data types specifically
    if isinstance(values, np.ndarray):
        if values.dtype == np.float32 or values.dtype == np.float64:
            return torch.tensor(values, dtype=torch.float32)
        elif values.dtype == np.int32 or values.dtype == np.int64:
            return torch.tensor(values, dtype=torch.int64)
        else:
            return torch.tensor(values)  # This covers other data types like boolean, etc.
    
    # For lists or other iterable types, check the data type of the first element
    if isinstance(values, (list, tuple)):
        if values:  # Check if the list/tuple is not empty
            first_element = values[0]
            if isinstance(first_element, float):
                return torch.tensor(values, dtype=torch.float32)
            elif isinstance(first_element, int):
                return torch.tensor(values, dtype=torch.int64)
            elif isinstance(first_element, bool):
                return torch.tensor(values, dtype=torch.bool)
            else:
                return torch.tensor(values)  # Default case for other types
        else:
            raise ValueError("Empty list or tuple provided; cannot determine tensor dtype.")

    # Catch-all for other data types that are directly convertible
    try:
        return torch.tensor(values)
    except ValueError as e:
        raise ValueError(f"Unsupported data type for tensor conversion: {type(values)}. Error: {e}")

def get_test_results(inferred_y, target_y, task_type, num_classes, average='macro'):
    """
    Calculates performance metrics for tasks using PyTorch tensors that might have batch and sequence dimensions.
    Parameters:
    - inferred_y: Predictions from the model (as PyTorch tensors).
    - target_y: Ground truth labels or values (as PyTorch tensors).
    - padding_mask: Mask indicating valid data points (1 for valid and 0 for padding).
    - task_type: Type of task (e.g., 'binary_classification', 'multi_class_classification', 'multi_label_classification', 'regression').
    - num_classes: Number of classes for classification tasks. Not used for binary.
    - average: Averaging method for precision, recall, and F1 score.
    Returns:
    - metrics: A dictionary containing relevant performance metrics.
    """
    metrics = {}
    
    # Check inferred_y is tensor
    if not isinstance(inferred_y, torch.Tensor):
        inferred_y = convert_to_tensor(inferred_y)
        print("inferred_y is not tensor")
    if not isinstance(target_y, torch.Tensor):
        target_y = convert_to_tensor(target_y)
        
    # Move tensors to CPU for compatibility with sklearn metrics
    inferred_y = inferred_y.cpu()
    target_y = target_y.cpu()

    if task_type in ['binary_classification', 'multi_class_classification', 'ordinal_regression']:

        inferred_y = inferred_y.flatten()
        target_y = target_y.flatten()
        
        correct = (inferred_y == target_y).float().sum()
        accuracy = correct / target_y.numel()
        inferred_y_np = inferred_y.numpy()
        target_y_np = target_y.numpy()
        
        metrics['accuracy'] = accuracy.item()
        metrics['precision'] = precision_score(target_y_np, inferred_y_np, average=average, labels=range(num_classes), zero_division=0)
        metrics['recall'] = recall_score(target_y_np, inferred_y_np, average=average, labels=range(num_classes), zero_division=0)
        metrics['f1_score'] = f1_score(target_y_np, inferred_y_np, average=average, labels=range(num_classes), zero_division=0)
        if task_type == 'binary_classification':
            metrics['roc_auc'] = roc_auc_score(target_y_np, inferred_y_np)
           
    elif task_type == 'multi_label_classification':
        inferred_y = (inferred_y > 0.5).float()
        accuracy = accuracy_score(target_y.numpy().reshape(-1), inferred_y.numpy().reshape(-1), normalize=True)
        metrics['accuracy'] = accuracy

    elif task_type in ['regression', 'compositional_regression']:
        mse = torch.mean((target_y - inferred_y) ** 2)
        mae = torch.mean(torch.abs(target_y - inferred_y))
        ss_total = torch.sum((target_y - torch.mean(target_y)) ** 2)
        ss_res = torch.sum((target_y - inferred_y) ** 2)
        if ss_total == 0:
            r2 = torch.tensor(float('nan'))
        else:
            r2 = 1 - ss_res / ss_total

        metrics['mse'] = mse.item()
        metrics['mae'] = mae.item()
        metrics['r2'] = r2.item()

    return metrics
    