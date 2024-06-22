import torch
import numpy as np

def convert_explanation_to_image_shape(explanation, image_shape, explain_size, image_elements):
    """
    Convert the explanation vector to match the target image shape with the first dimension set to 1.
    
    Args:
        explanation (torch.Tensor): The explanation vector of shape [batch_size, explain_size].
        image_shape (tuple): The shape of the target image, e.g., (channels, height, width).
        explain_size (int): The size of the explanation vector.

    Returns:
        torch.Tensor: Explanation vector expanded and reshaped to match the image shape.
    """
    # Set first dim to 1, rest match target shape
    explain_shape = [1] + list(image_shape[1:])
    
    # Repeat the explanation to match the total volume of the target image shape
    repeat_times = image_elements // explain_size
    e1 = explanation.repeat(1, repeat_times)
    
    # Handle any remaining elements if the image elements aren't perfectly divisible by explain_size
    remaining_elements = image_elements % explain_size
    e2 = explanation[:, :remaining_elements]
    
    # Concatenate the repeated and zero-padded parts
    expanded_e = torch.cat([e1, e2], dim=-1)
    
    # Reshape the explanation vector to match the new shape
    expanded_e = expanded_e.view(-1, *explain_shape)
    
    return expanded_e

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
            # If it's a list, tuple, or torch.Size, take the last dimension
            if isinstance(shape[-1], (list, tuple)):
                raise ValueError(f"Unsupported nested shape type: {type(shape)}")
            else:
                return shape[-1]
        elif isinstance(shape, (int, np.integer)):
            # Handle both Python and numpy integers
            return int(shape)
        else:
            raise ValueError(f"Unsupported shape type: {type(shape)}")

    if isinstance(feature_shapes, (list, tuple)):
        for nf in feature_shapes:
            feature_sizes.append(extract_size(nf))
    else:
        return extract_size(feature_shapes)
    
    return feature_sizes