import torch
import torch.nn.functional as F

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
    e2 = torch.zeros_like(explanation[:, :remaining_elements])
    
    # Concatenate the repeated and zero-padded parts
    expanded_e = torch.cat([e1, e2], dim=-1)
    
    # Reshape the explanation vector to match the new shape
    expanded_e = expanded_e.view(-1, *explain_shape)
    
    return expanded_e

def determine_activation_by_task_type(task_type):
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
    if task_type == "multi_class_classification":
        return 'softmax'
    elif task_type == "binary_classification":
        return 'sigmoid'
    elif task_type == "multi_label_classification":
        return 'sigmoid'  # Multiple independent binary classifications
    elif task_type == "regression":
        return 'linear'  # Typically no activation function (i.e., linear) is used for regression outputs
    else:
        raise ValueError(f"Invalid task type: {task_type}")

def generate_condition_data(label_shape, task_type, device, enable_discrete_conditions=False):
    """
    Generates task-specific condition data for different types of machine learning tasks,
    ensuring the condition data matches the label shape and appropriate data type.

    Args:
    - label_shape (tuple): The shape of the output tensor expected to match label specifications.
    - task_type (str): Specifies the machine learning task type.
    - device (str): Specifies the device for tensor operations.

    Returns:
    - Tensor: A tensor of condition data appropriate for the specified task type, all in float dtype.
    """
    logits = torch.randn(label_shape).to(device)
    if task_type == "multi_class_classification":
        # Generate indices and convert to one-hot encoding to maintain dimensionality
        class_indices = torch.argmax(logits, dim=-1)
        condition_data = F.one_hot(class_indices, num_classes=logits.shape[-1])
    elif task_type in ["binary_classification", "multi_label_classification"]:
        # Generate binary labels and convert to float
        condition_data = (torch.sigmoid(logits) > 0.5)
    elif task_type == "regression":
        # Directly use continuous values for regression
        condition_data = logits
    else:
        # Use random noise for unknown task types, ensuring it's in float dtype
        condition_data = logits

    return condition_data.float()

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
    if task_type == "multi_class_classification":
        logits = torch.randn(label_shape).to(device)
        # Use softmax to simulate probabilities across classes
        condition_data = torch.softmax(logits, dim=-1)
    elif task_type in ["binary_classification", "multi_label_classification"]:
        condition_data = torch.rand(label_shape).to(device)
    elif task_type == "regression":
        # For regression tasks, always use continuous values
        condition_data = torch.randn(label_shape).to(device)
    else:
        # For unknown task types, generate random noise
        condition_data = torch.randn(label_shape).to(device)

    # Ensure the condition data is in float dtype unless explicitly discrete
    return condition_data.float()
