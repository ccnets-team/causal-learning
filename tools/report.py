'''
Author:
        
        PARK, JunHo, junho@ccnets.org
        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import numpy as np

def calculate_test_results(inferred_y, target_y, task_type = None, num_classes=None, average='macro'):
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

    # Move tensors to CPU for compatibility with sklearn metrics
    inferred_y = inferred_y.cpu()
    target_y = target_y.cpu()

    if task_type in ['binary_classification', 'multi_class_classification', 'ordinal_regression']:
        if task_type == 'binary_classification':
            inferred_y = (inferred_y > 0.5).float()
        elif task_type == 'multi_class_classification':
            inferred_y = torch.argmax(inferred_y, dim=-1)
            target_y = torch.argmax(target_y, dim=-1)  # Assuming one-hot encoding of target
        elif task_type == "ordinal_regression":
            inferred_y = torch.round(inferred_y).int().clamp(0, num_classes - 1)
            target_y = torch.round(target_y).int()

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
