'''
Author:
        
        PARK, JunHo, junho@ccnets.org
        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import numpy as np

def calculate_test_results(inferred_y, target_y, padding_mask = None, task_type = None , num_classes=None, average='macro'):
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

    # Apply the padding mask if provided
    if padding_mask is not None:
        # Flatten the mask and use it to filter out padded values
        valid_indices = padding_mask.bool().expand_as(inferred_y)
        label_size = inferred_y.size(-1)
        inferred_y = inferred_y[valid_indices].view(-1, label_size)
        target_y = target_y[valid_indices].view(-1, label_size)

    # Move tensors to CPU for compatibility with sklearn metrics
    inferred_y = inferred_y.cpu()
    target_y = target_y.cpu()

    if task_type in ['binary_classification', 'multi_class_classification']:
        if task_type == 'binary_classification':
            inferred_y = (inferred_y > 0.5).float()
        elif task_type == 'multi_class_classification':
            inferred_y = torch.argmax(inferred_y, dim=-1)
            target_y = torch.argmax(target_y, dim=-1)  # Assuming one-hot encoding of target

        correct = (inferred_y == target_y).float().sum()
        accuracy = correct / target_y.numel()
        inferred_y_np = inferred_y.numpy()
        target_y_np = target_y.numpy()
        
        metrics['accuracy'] = accuracy.item()
        metrics['precision'] = precision_score(target_y_np, inferred_y_np, average=average, labels=range(num_classes), zero_division=0)
        metrics['recall'] = recall_score(target_y_np, inferred_y_np, average=average, labels=range(num_classes), zero_division=0)
        metrics['f1_score'] = f1_score(target_y_np, inferred_y_np, average=average, labels=range(num_classes), zero_division=0)
        
    elif task_type == 'multi_label_classification':
        inferred_y = (inferred_y > 0.5).float()
        accuracy = accuracy_score(target_y.numpy().reshape(-1), inferred_y.numpy().reshape(-1), normalize=True)
        metrics['accuracy'] = accuracy

    elif task_type == 'regression':
        mse = torch.mean((target_y - inferred_y) ** 2)
        mae = torch.mean(torch.abs(target_y - inferred_y))
        ss_total = torch.sum((target_y - torch.mean(target_y)) ** 2)
        ss_res = torch.sum((target_y - inferred_y) ** 2)
        r2 = 1 - ss_res / ss_total if ss_total > 0 else torch.tensor(np.inf)

        metrics['mse'] = mse.item()
        metrics['mae'] = mae.item()
        metrics['r2'] = r2.item()

    return metrics
