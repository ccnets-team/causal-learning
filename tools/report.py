'''
Author:
        
        PARK, JunHo, junho@ccnets.org
        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_test_results(inferred_y, target_y, task_type, label_size):
    """
    Calculates performance metrics for binary classification, multiclass classification, 
    multi-label classification, and regression tasks using PyTorch.
    Parameters:
    - inferred_y: Predictions from the model (as PyTorch tensors).
    - target_y: Ground truth labels or values (as PyTorch tensors).
    - task_type: Type of task ('binary_classification', 'multi_class_classification', 
                 'multi_label_classification', 'regression').
    - label_size: Number of classes for classification tasks. Not used for binary.
    Returns:
    - metrics: A dictionary containing relevant performance metrics.
    """
    metrics = {}
    inferred_y = inferred_y.cpu()
    target_y = target_y.cpu()

    if task_type in ['binary_classification', 'multi_class_classification']:
        if task_type == 'binary_classification':
            inferred_y = (inferred_y > 0.5).float()
        elif task_type == 'multi_class_classification':
            inferred_y = torch.argmax(inferred_y, dim=-1)

        correct = (inferred_y == target_y).float().sum()
        accuracy = correct / target_y.shape[0]
        metrics['accuracy'] = accuracy.item()

        inferred_y_np = inferred_y.numpy()
        target_y_np = target_y.numpy()

        metrics['precision'] = precision_score(target_y_np, inferred_y_np, average='macro')
        metrics['recall'] = recall_score(target_y_np, inferred_y_np, average='macro')
        metrics['f1_score'] = f1_score(target_y_np, inferred_y_np, average='macro')

    elif task_type == 'multi_label_classification':
        # Assuming threshold of 0.5 for binary relevance per label
        inferred_y = (inferred_y > 0.5).float()

        accuracy = accuracy_score(target_y.numpy(), inferred_y.numpy(), normalize=True)
        metrics['accuracy'] = accuracy

        # Other relevant metrics like Hamming loss can be added here

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
