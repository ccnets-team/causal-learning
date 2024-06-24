import torch
from torch import nn
import numpy as np

class FeatureToImageShape(nn.Module):
    def __init__(self, input_size, image_shape):
        super(FeatureToImageShape, self).__init__()
        self.input_size = input_size
        self.image_shape = image_shape
        self.image_elements = torch.prod(torch.tensor(image_shape[1:])).item()
        self.feature_shape = (1,) + image_shape[1:]
        
        # Calculate how many times to repeat the feature to match the target image size
        self.repeat_times = self.image_elements // self.input_size
        self.remaining_elements = self.image_elements % self.input_size
        
    def forward(self, feature):
        """
        Convert the feature vector to match the target image shape with the first dimension set to 1.
        
        Args:
            feature (torch.Tensor): The feature vector of shape [batch_size, input_size].

        Returns:
            torch.Tensor: Feature vector expanded and reshaped to match the image shape.
        """
        batch_size = feature.size(0)

        # Repeat and handle any remaining elements
        expanded_feature = feature.repeat_interleave(self.repeat_times, dim=-1)
        if self.remaining_elements > 0:
            expanded_feature = torch.cat([expanded_feature, feature[:,...,:self.remaining_elements]], dim=-1)
        
        # Reshape the feature vector to match the target image shape
        reshaped_feature = expanded_feature.view(batch_size, *self.feature_shape)
        
        return reshaped_feature
    
class FlipTensor(nn.Module):
    def __init__(self, output_shape):
        super(FlipTensor, self).__init__()
        self.use_image = len(output_shape) != 1

    def forward(self, tensor, padding_mask=None):
        """
        Reverses the order of elements in the tensor along the specified dimension.

        Parameters:
            tensor (Tensor): The tensor to be reversed.
            padding_mask (Tensor, optional): Optional padding mask to reverse.

        Returns:
            Tuple[Tensor, Tensor]: The reversed tensor and the reversed padding mask.
        """
        if self.use_image:
            return tensor, padding_mask
        
        reversed_tensor = torch.flip(tensor, dims=[1])
        if padding_mask is not None:
            reversed_padding_mask = torch.flip(padding_mask, dims=[1])
            return reversed_tensor, reversed_padding_mask
        return reversed_tensor, None

def determine_activation_function(task_type):
    """
    Determines the appropriate activation function for a model component based on the machine learning task type.

    Parameters:
    - task_type (str): Specifies the type of machine learning task. Valid types are:
        'binary_classification', 'multi_class_classification', 'multi_label_classification', 'regression'.

    Returns:
    - str: The name of the activation function suitable for the given task type. 'linear' indicates no activation.

    Raises:
    - ValueError: If an unsupported task type is provided.
    """
    if task_type in ["binary_classification"]: # binary classification is one-hot encoded
        return 'sigmoid'
    elif task_type in ["multi_class_classification", "compositional_regression"]: # binary classification is one-hot encoded
        return 'softmax'
    elif task_type in ["multi_label_classification"]:
        return 'sigmoid'  # Multiple independent binary classifications
    elif task_type in ["regression", "ordinal_regression"]:
        return 'linear'  # Typically no activation function (i.e., linear) is used for regression outputs
    else:
        raise ValueError(f"Invalid task type: {task_type}")

def generate_condition_data(label_shape, task_type, device):
    """
    Generates task-specific condition data for different types of machine learning tasks,
    ensuring the condition data matches the label shape and appropriate data type.

    Args:
    - label_shape (tuple): The shape of the output tensor expected to match label specifications.
    - task_type (str): Specifies the machine learning task type.
    - device (str): Specifies the device for tensor operations.
    - enable_discrete_conditions (bool): If True, generates discrete condition data for applicable tasks.

    Returns:
    - Tensor: A tensor of condition data appropriate for the specified task type, all in float dtype.
    """
    if task_type in ["binary_classification"]:
        condition_data = torch.rand(label_shape).to(device)
        condition_data = (condition_data > 0.5).float()
    elif task_type in ["multi_class_classification", "compositional_regression"]: 
        logits = torch.randn(label_shape).to(device)
        # Use softmax to simulate probabilities across classes
        condition_data = torch.softmax(logits, dim=-1)
        # Pick one in the form of one-hot encoding
        condition_data = torch.zeros_like(condition_data).scatter_(-1, torch.argmax(condition_data, dim=-1, keepdim=True), 1)
    elif task_type in ["multi_label_classification"]:
        condition_data = torch.rand(label_shape).to(device)
        condition_data = (condition_data > 0.5).float()
    elif task_type in ["regression", "ordinal_regression"]:
        # For regression tasks, always use continuous values
        condition_data = torch.randn(label_shape).to(device)
    else:
        # For unknown task types, generate random noise
        condition_data = torch.randn(label_shape).to(device)

    # Ensure the condition data is in float dtype unless explicitly discrete
    return condition_data.float()

def extend_obs_shape_channel(obs_shape):
    return [obs_shape[0] + 1] + list(obs_shape[1:])

def reduce_tensor(input_tensor, padding_mask, dim):
    """
    Reduce the tensor by computing the mean or sum, considering only non-padded data if a padding mask is provided.

    Args:
        input_tensor (Tensor): The input tensor to reduce.
        padding_mask (Tensor, optional): The mask indicating padded elements to exclude from the calculation.
        dim (int): The dimension along which to reduce the tensor.

    Returns:
        Tensor: The reduced tensor.
    """
    if padding_mask is not None:
        # Apply the padding mask to the input tensor
        input_tensor *= padding_mask
        expanded_mask = padding_mask.expand_as(input_tensor)
        # Compute the sum of the input tensor, considering only the non-padded data
        reduced_tensor = input_tensor.sum(dim=dim, keepdim=True) / expanded_mask.sum(dim=dim, keepdim=True).clamp_min(1)
    else:
        # Compute the mean of the input tensor
        reduced_tensor = input_tensor.mean(dim=dim, keepdim=True)
    
    return reduced_tensor

def convert_shape_to_size(feature_shapes):
    """
    Convert feature shapes to their respective sizes.
    
    Parameters:
    - feature_shapes (list or tuple): A list or tuple of feature shapes. Each shape can be an integer, 
      a list of integers, a torch.Size object, or a nested list.

    Returns:
    - list: A list of sizes corresponding to the feature shapes.
    """
    feature_sizes = []

    def extract_size(shape):
        """Extract the size from various types of shape representations."""
        if isinstance(shape, (list, tuple, torch.Size)):
            if len(shape) == 1:
                return shape[-1]
            else:
                return list(shape)
        elif isinstance(shape, (int, np.integer)):
            # Handle both Python and numpy integers
            return int(shape)
        else:
            raise ValueError(f"Unsupported shape type: {type(shape)}")

    if isinstance(feature_shapes, (list, tuple)):
        for shape in feature_shapes:
            feature_sizes.append(extract_size(shape))
    else:
        return extract_size(feature_shapes)
    
    return feature_sizes

def find_image_indices(shapes):
    image_indices = []
            
    if isinstance(shapes, (list, tuple)):
        for idx, shape in enumerate(shapes):
            if isinstance(shape, (list, tuple)) and len(shape) == 3:
                image_indices.append(idx)
    return image_indices