'''-ResNet in PyTorch.

Reference:
[1] Modified by PARK, JunHo in April 10, 2022

[2] Writed by Jinsu, Kim in August 11, 2022

[3] Writed by JeongYoong, Kim in April 24, 2024

COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''

DEFAULT_PRINT_INTERVAL = 100

import pandas as pd
import numpy as np
from IPython.display import display
from tools.config.ml_config import MLConfig
from tools.config.data_config import DataConfig

def print_iter(epoch, num_epoch, iters, len_dataloader, et):
    print('[%d/%d][%d/%d][Time %.2f]'
        % (epoch, num_epoch, iters, len_dataloader, et))

def print_trainer(trainer_type, trainer_name, metrics):
    losses = metrics.losses
    errors = metrics.errors
    print(trainer_type + ": ", "Three " + trainer_name)
    print('Inf: %.4f\tGen: %.4f\tRec: %.4f\tE: %.4f\tR: %.4f\tP: %.4f'
        % (losses['inference_loss'], losses['generation_loss'], losses['reconstruction_loss'], errors['explainer_error'], errors['reasoner_error'], errors['producer_error']))

def print_lr(*_optimizers):
    lr_list = []
    for optimizers_group in _optimizers:
        for optimizer in optimizers_group:
            lr_list.append(optimizer.param_groups[0]['lr'])
    
    # Check if all LRs in the list are the same
    if all(lr == lr_list[0] for lr in lr_list):
        print(f'Unified LR across all optimizers: {lr_list[0]}')
    else:
        print('Not all optimizers have synchronized LRs.')
        for optimizers_group in _optimizers:
            for optimizer in optimizers_group:
                print(f'Opt-{type(optimizer).__name__} LR: {optimizer.param_groups[0]["lr"]}')

def print_results(metrics, is_eval=False):
    """
        Create and print string of test metrics.
        
        Parameters
        ---------
        label_types : Iterable obj of str
            Possible values:
                - 'precision'
                - 'recall'
                - 'f1'
                - 'mse'
                - 'mae'
                - 'r2'
                - 'auc'
                - 'log_loss'
            The type of metrics used for evaluation.
        metrics : list os float
            A list containing individual metrics values as floats.

        Returns
        -------
            txt : str
                Returns a string containing metrics information.
    """
    if is_eval==False:
        print()
    else:
        print('=====================Eval Metrics========================')

    for label_type, metric in metrics.items():
        if np.isscalar(metric):
            text = (f'{label_type}: %.4f'
                % metric)
            print(text)

    print()
    
def print_checkpoint_info(parent, time_cost, epoch_idx, iter_idx, len_dataloader, 
                            ccnet_metric = None, train_metrics=None):
    """Prints formatted information about the current checkpoint."""
    ccnet = parent.ccnet
    trainer = parent.trainer
        
    print_iter(epoch_idx, parent.num_epoch, iter_idx, len_dataloader, time_cost)
    if ccnet_metric is not None:
        print_lr(trainer.optimizers)
    print('=====================Train Metrics=======================')
    
    if ccnet_metric is not None:
        ccnet_type = "CCNet" 
        ccnet_name = ccnet.model_name.capitalize()
        print_trainer(ccnet_type, ccnet_name, ccnet_metric)
    if train_metrics is not None:
        print_results(train_metrics, is_eval=False)

# Main function to print all configurations including ML parameters and data configurations.
def print_ml_params(trainer_name, ml_config: MLConfig, data_config: DataConfig):
    print("Trainer Name:", trainer_name)
    print("\n")

    # Print all ML parameters
    print_parameters(ml_config)
    
    # Print data configuration details
    print_dataconfig(data_config)
    
    print("\n")  # Print a newline for better readability at the end

# Function to print parameters of ML models. Iterates through all parameters in the list.
def print_parameters(ml_param_list):
    # Iterate through each parameter in the ML parameters list
    for param in ml_param_list:
        # Extract and organize parameter data into a main and sub-dataFrames
        main_data, sub_data = extract_parameter_as_dataframe(param)
        # Print the summary of parameters in bold
        print(f"\033[1m{param.__class__.__name__} Parameters:\033[0m")
        display(main_data)  # Display the main DataFrame using IPython.display for better formatting
        
        # Iterate through sub-data and display each in detail with italic formatting
        for key, df in sub_data.items():
            # Remove 'y_dim' and 'e_dim' columns to prevent repeat if they exist 
            filtered_df = df.loc[:, ~df.columns.str.contains('y_dim|e_dim')]
                
            print(f"\033[3m\nDetailed {key} Configuration:\033[0m")
            display(filtered_df)  # Display each sub-dataFrame
        # print("\n")

# Function to extract data from parameters. It handles user-defined classes and standard data types.
def extract_parameter_as_dataframe(param):
    if param is None:
        return pd.DataFrame(), {}

    main_data = {}
    sub_data = {}
    # Use dir() to get all attributes of the parameter and filter them
    for key in dir(param):  
        value = getattr(param, key, None)
        if not key.startswith('__') and not callable(value):
            if hasattr(value, '__dict__'):  
                # For user-defined class instances, point to further details below
                main_data[key] = "See details below"
                # Create detailed DataFrame for user-defined class instances
                sub_data[key] = pd.DataFrame([{f"{key}_{subkey}": subvalue for subkey, subvalue in vars(value).items()}])
            else:
                main_data[key] = [value]

    main_df = pd.DataFrame(main_data)
    return main_df, sub_data

# Function to extract and format data from the DataConfig object into a DataFrame.
def extract_dataconfig_as_dataframe(data_config):
    data = {
        "dataset_name": [data_config.dataset_name],
        "task_type": [data_config.task_type],
        "obs_shape": [data_config.obs_shape],
        "label_size": [data_config.label_size],
        "explain_size": [data_config.explain_size],
        "show_image_indices": [data_config.show_image_indices]
    }
    return pd.DataFrame(data)

# Function to print data configuration parameters.
def print_dataconfig(data_config):
    # Extract and format data configuration into a DataFrame
    df = extract_dataconfig_as_dataframe(data_config)
    # Print the data configuration parameters in bold
    print("\033[1mDataConfig Parameters:\033[0m")
    
    display(df)  # Use IPython.display to render the DataFrame cleanly
    print("\n")


# Function to print CCNetAnalysis results.
def print_analysis_result(task_type, label_size, metrics, label_scale=None):
    label_names = list(range(label_size))
    label_names = [str(num) for num in label_names]

    if 'confusion_matrix' in metrics:
        conf_matrix = metrics.pop('confusion_matrix')
    else:
        conf_matrix = [[0, 0], [0, 0]] # Default confusion matrix 

    # 최대 길이 계산
    max_metric_key_length = max(len(key) for key in metrics.keys())
    max_value_length = max(len(f"{value:.4f}") for value in metrics.values())
    metric_header_length = max_metric_key_length + max_value_length + 3
    matrix_header_length = 3 * len(label_names) * 5  
    
    # 헤더 줄
    print("Analysis Results")
    print("Task Type:", task_type)
    print("Label Size:", label_size)
    print("=" * metric_header_length + "=" * 5 + "=" * matrix_header_length)
    print()
    print("Performance Metrics".ljust(metric_header_length) + " " * 5 + "Confusion Matrix".center(matrix_header_length))
    print("=" * metric_header_length + " " * 5 + "=" * matrix_header_length)
    print(f"{'Metric':<{max_metric_key_length}} | {'Value':>{max_value_length}}", end=" " * 5)
    table_header = 'True\Pred'
    print(f"{table_header:>{10}} | {' '.join([name.center(10) for name in label_names])}")
    print("-" * metric_header_length + " " * 5 + "-" * matrix_header_length)

    if task_type == 'binary_classification' or task_type == 'multi_class_classification':
        metrics_list = list(metrics.items())
        for i in range(max(len(metrics_list), len(conf_matrix))):
            metric_line = " " * metric_header_length
            conf_line = " " * matrix_header_length

            if i < len(metrics_list):
                metric, value = metrics_list[i]
                metric_line = f"{metric:<{max_metric_key_length}} | {value:>{max_value_length}.4f}"

            if i < len(conf_matrix):
                conf_line = f"{label_names[i]:>{10}} | {' '.join(f'{num:^10}' for num in conf_matrix[i])}"

            print(metric_line + " " * 5 + conf_line)

        print("=" * metric_header_length + "=" * 5 + "=" * matrix_header_length)

