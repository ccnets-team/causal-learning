import torch

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
