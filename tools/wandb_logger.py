'''
Author:
        
        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

try:
    import wandb
except ImportError:
    wandb = None
from datetime import datetime
import os

now = datetime.now()
formatted_date = now.strftime("%y-%m-%d %H:%M:%S")

def sort_key(item):
    key, value = item
    if isinstance(value, str):
        return (0, key)  
    elif isinstance(value, bool):
        return (1, key)  
    else:
        return (2, key)  

# Added helper function to remove specified fields
def remove_fields(d, fields):
    if d is None:
        return None
    return {k: v for k, v in d.items() if k not in fields}

METRICS_CATEGORY_MAP = {
    'losses': 'Losses',
    'errors': 'Errors'
}
# Conversion function to turn nested dataclasses into dictionaries
def convert_to_dict(obj):
    """Convert a dataclass object to a dictionary, including nested objects."""
    if obj is not None:
        params_dict = obj.__dict__.copy()
        for key, value in params_dict.items():
            if hasattr(value, '__dict__'):
                params_dict[key] = convert_to_dict(value)
        return params_dict

def rename_this_function(data_config, name_prefix=None):
    """Convert a dataclass to a dictionary, filter out non-primitive types, and sort it."""
    data_config_dict = convert_to_dict(data_config)
    data_config_dict = {k: v for k, v in data_config_dict.items() if isinstance(v, (int, float, str, bool))}
    data_config_dict = dict(sorted(data_config_dict.items()))
    if name_prefix:
        data_config_dict = {name_prefix: data_config_dict}
    return data_config_dict

def wandb_init(data_config, ml_params):
    if wandb is None:
        raise RuntimeError("wandb is not installed. Please install wandb to use wandb_init.")
    wandb.login()
    
    data_config_dict = rename_this_function(data_config, 'data')
    model_params_dict = rename_this_function(ml_params.model, 'model')
    optimization_params_dict = rename_this_function(ml_params.optimization, 'optimization')
    training_params_dict = rename_this_function(ml_params.training, 'training')
    
    # Remove duplicate variables from model_params_dict
    for key in ['obs_shape', 'y_dim', 'e_dim']:
        model_params_dict['model'].pop(key, None)
    
    merged_config_dict = {**data_config_dict, **model_params_dict, **optimization_params_dict, **training_params_dict}
    
    trainer_name = 'causal-learning'
    
    wandb.init(
        project=trainer_name,
        name=f'{data_config.dataset_name} : {formatted_date}',
        save_code=False,
        monitor_gym=False, 
        config=merged_config_dict
    )
    
    directory_path = f'../saved/{data_config.dataset_name}/{trainer_name}'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    artifact = wandb.Artifact(f'{trainer_name}-{data_config.dataset_name}', type='model')
    artifact.add_dir(directory_path, name="saved/")
    wandb.log_artifact(artifact)
    
def wandb_end():
    if wandb is None:
        raise RuntimeError("wandb is not installed. Please install wandb to use wandb_end.")
    wandb.finish()

def _wandb_log_data(metrics, log_data = None, iters = None):
    if wandb is None:
        print("wandb is not installed. Skipping wandb_log_data.")
        return

    if metrics is not None :
        # Loop over each metrics category and log each metric with a specific prefix
        for category_name, category_metrics in metrics.items():
            # Map the category_name to the new desired name
            mapped_category_name = METRICS_CATEGORY_MAP.get(category_name, category_name.title())
            
            for metric_name, metric_value in category_metrics.items():
                if metric_value is not None:
                    components = metric_name.split('_')
                    new_metric_name = components[-2].title() if len(components) > 1 else components[0].title()                         
                    log_name = f"{mapped_category_name}/{new_metric_name}"
                    log_data[log_name] = metric_value  # Add the metric to the logging dictionary
    else:
        ValueError("No metrics to log.")
        return
    wandb.log(log_data, step=iters)  # Log all data including the metrics

def wandb_log_train_metrics(time_cost, lr, ccnet_metric=None, images=None, iters = None):
    # Define step_logs as a dictionary with relevant key-value pairs
    
    if ccnet_metric is not None:
        gpt_ccnet_losses = dict(ccnet_metric.losses.data)
        gpt_ccnet_errors = dict(ccnet_metric.errors.data)
        
        ccnet_metric = {
        'CCNet/Losses': gpt_ccnet_losses,
        'CCNet/Errors': gpt_ccnet_errors,
        }
    else:
        ccnet_metric = {
        }
    
    additional_logs = {"Step/LearningRate": lr, 
                       "Step/TimeCost": time_cost
                       }
    if images is not None:
        additional_logs["WB Images"] = images
    
    log_data = {**additional_logs}
    train_metrics = {**ccnet_metric}
    _wandb_log_data(train_metrics, log_data, iters = iters)

def wandb_log_train_data(metrics, images, iters):
    additional_logs = {}
    if images is not None:
        additional_logs["WB Images"] = images
        
    log_data = {**additional_logs}
    train_metrics = {"Training": metrics}
    _wandb_log_data(train_metrics, log_data=log_data, iters = iters)
    
def wandb_log_eval_data(metrics, images, iters):
    additional_logs = {}
    if images is not None:
        additional_logs["WB Images"] = images
        
    log_data = {**additional_logs}
    eval_metrics = {"Evaluate": metrics}
    _wandb_log_data(eval_metrics, log_data=log_data, iters = iters)

def wandb_log_test_data(metrics, iters, images=None):
    additional_logs = {}
    if images is not None:
        additional_logs["WB Images"] = images
        
    log_data = {**additional_logs}
    test_metrics = {"Test": metrics}
    _wandb_log_data(test_metrics, log_data=log_data, iters = iters)

def wandb_image(image):
    return wandb.Image(image)
