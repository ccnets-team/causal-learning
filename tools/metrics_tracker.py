'''
Reference:
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''

import torch
from collections import OrderedDict
import numpy as np

def convert_to_float(kwargs, keys_list):
    def process_value(value):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if len(value.shape) >= 2:
                value = value[:, 0, ...]  # Extract the first index of sequence
            return float(value.mean())
        else:
            return value

    return OrderedDict(
        (key, process_value(value))
        for key, value in kwargs.items() if key in keys_list
    )

class MetricsBase:
    def __init__(self, data, keys_list):
        self.data = convert_to_float(data, keys_list)

    def items(self):
        return self.data.items()
            
    def __iadd__(self, other):
        all_keys = list(self.data.keys()) + [key for key in other.data.keys() if key not in self.data]  # Maintain order
        for key in all_keys:
            self_val = self.data.get(key)
            other_val = other.data.get(key)
            
            if self_val is None and other_val is not None:
                self.data[key] = other_val  # Maintain order when adding new items
            elif self_val is not None and other_val is not None:
                self.data[key] += other_val
        return self

    def __itruediv__(self, divisor):
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        for key, value in self.data.items():
            if value is not None:
                self.data[key] /= divisor
        return self
    
    def __iter__(self):
        yield from self.data.values()

class CoopErrorMetrics(MetricsBase):
    def __init__(self, **kwargs):
        super().__init__(kwargs, ['explainer_error', 'reasoner_error', 'producer_error'])
        
    def __getitem__(self, key):
        return self.data[key]
        
class PredictionLossMetrics(MetricsBase):
    def __init__(self, **kwargs):
        super().__init__(kwargs, ['inference_loss', 'generation_loss', 'reconstruction_loss'])
    
    def __getitem__(self, key):
        return self.data[key]
    
class MetricsTracker:
    def __init__(self, 
                 losses: PredictionLossMetrics=None,
                 errors: CoopErrorMetrics=None 
                 ):

        self.losses = losses or PredictionLossMetrics()
        self.errors = errors or CoopErrorMetrics()
    
    def reset(self):
        self.losses = PredictionLossMetrics()
        self.errors = CoopErrorMetrics()
        
    def __iadd__(self, other):
        self.losses += other.losses
        self.errors += other.errors
        return self

    def __truediv__(self, divisor):
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        new_losses = PredictionLossMetrics(**{k: v / divisor for k, v in self.losses.items()})
        new_errors = CoopErrorMetrics(**{k: v / divisor for k, v in self.errors.items()})
        return MetricsTracker(losses=new_losses, errors=new_errors)
    
    def __iter__(self):
        yield from self.losses.items()
        yield from self.errors.items()

def create_causal_training_metrics(**kwargs):
    """Helper method to create TrainingMetrics object."""

    def compute_masked_mean(tensor, mask = None):
        if tensor is None:
            return None
        if mask is None:
            return tensor.mean()
        else:
            reduced_tensor = tensor.mean(dim=-1, keepdim = True)
            if reduced_tensor.shape == mask.shape:
                return reduced_tensor[mask > 0].mean() 
            else: 
                return reduced_tensor.mean()

    padding_mask = kwargs.get('padding_mask')
    
    errors = CoopErrorMetrics(
        explainer_error= compute_masked_mean(kwargs.get('explainer_error'), padding_mask),
        reasoner_error= compute_masked_mean(kwargs.get('reasoner_error'), padding_mask),
        producer_error= compute_masked_mean(kwargs.get('producer_error'), padding_mask)
    )
    
    losses = PredictionLossMetrics(
        inference_loss= compute_masked_mean(kwargs.get('inference_loss'), padding_mask),
        generation_loss= compute_masked_mean(kwargs.get('generation_loss'), padding_mask),
        reconstruction_loss= compute_masked_mean(kwargs.get('reconstruction_loss'), padding_mask)
    )

    return MetricsTracker(
        losses=losses,
        errors=errors
    )
    