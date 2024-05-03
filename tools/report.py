'''
Author:
        
        PARK, JunHo, junho@ccnets.org
        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import numpy as np
import torch
from tools.print import print_metrics

def calculate_test_results(inferred_y, target_y, task_type, label_size):
    """
    Calculates performance metrics for binary classification, multiclass classification, and regression tasks using PyTorch.
    Parameters:
    - inferred_y: Predictions from the model (as PyTorch tensors).
    - target_y: Ground truth labels or values (as PyTorch tensors).
    - task_type: Type of task ('binary', 'classification', 'regression').
    - target_size: Number of classes for classification tasks. Not used for binary.
    - use_wandb: Whether to log metrics to Weights & Biases.
    Returns:
    - metrics: A dictionary containing relevant performance metrics.
    """
    metrics = {}
    
    if task_type in ['binary', 'classification']:
        if task_type == 'binary' and label_size == 2:
            inferred_y = (inferred_y > 0.5).long()
        else:
            inferred_y = torch.argmax(inferred_y, dim=-1)
            target_y = torch.argmax(target_y, dim=-1)
        
        accuracy, precision, recall, f1_score = calculate_multiclass_classification_metrics(inferred_y, target_y, num_classes = label_size)
        metrics['accuracy'] = accuracy
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1_score

    elif task_type == 'regression':
        mse = torch.mean((target_y - inferred_y) ** 2)
        mae = torch.mean(torch.abs(target_y - inferred_y))
        ss_total = torch.sum((target_y - torch.mean(target_y)) ** 2)
        ss_res = torch.sum((target_y - inferred_y) ** 2)
        r2 = 1 - ss_res / ss_total if ss_total > 0 else torch.tensor(np.inf)

        metrics['mse'] = mse.item()
        metrics['mae'] = mae.item()
        metrics['r2'] = r2.item()
    
    print_metrics(metrics)
        
    return metrics

def calculate_multiclass_classification_metrics(preds, labels, num_classes):
    """
    Calculate accuracy, precision, recall, and F1 score for multi-class classification using PyTorch.

    Args:
    preds (torch.Tensor): The predictions from the model (shape [batch_size, num_classes, 1])
    labels (torch.Tensor): The ground truth labels (shape [batch_size, 1, 1])

    Returns:
    dict: A dictionary containing accuracy, macro-precision, macro-recall, and macro-F1 score.
    """
    # Flatten and convert to class labels
    preds = preds.flatten()  # Predicted labels
    labels = labels.flatten()  # True labels

    # Calculate accuracy
    accuracy = torch.mean((preds == labels).float())

    # Initialize metrics
    precision_list = []
    recall_list = []
    f1_list = []

    # Calculate precision, recall, F1 for each class
    for c in range(num_classes):
        TP = torch.sum((preds == c) & (labels == c)).float()
        FP = torch.sum((preds == c) & (labels != c)).float()
        FN = torch.sum((preds != c) & (labels == c)).float()

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)

    # Calculate macro-averaged metrics
    macro_precision = torch.mean(torch.tensor(precision_list))
    macro_recall = torch.mean(torch.tensor(recall_list))
    macro_f1 = torch.mean(torch.tensor(f1_list))

    return (
        accuracy.item(),
        macro_precision.item(),
        macro_recall.item(),
        macro_f1.item()
    )