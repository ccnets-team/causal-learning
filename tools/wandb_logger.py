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

def convert_to_dict(ml_params):
    if ml_params is not None:
        params_dict = ml_params.__dict__.copy()

        for key, value in params_dict.items():
            if hasattr(value, '__dict__'):
                params_dict[key] = value.__dict__

        return params_dict

def sort_key(item):
    key, value = item
    if isinstance(value, str):
        return (0, key)  
    elif isinstance(value, bool):
        return (1, key)  
    else:
        return (2, key)  

METRICS_CATEGORY_MAP = {
    'losses': 'Losses',
    'errors': 'Errors'
}

def wandb_init(data_config, ml_params):
    if wandb is None:
        raise RuntimeError("wandb is not installed. Please install wandb to use wandb_init.")
    wandb.login()
    
    data_config_dict = convert_to_dict(data_config)
    ml_params_core_config, ml_params_encoder_config = convert_to_dict(ml_params.model.core_config),convert_to_dict(ml_params.model.encoder_config)
    ml_params_dict = convert_to_dict(ml_params)
    ml_params_dict['model']['core_config'] = ml_params_core_config
    ml_params_dict['model']['encoder_config'] = ml_params_encoder_config
    
    data_config_dict = {k: v for k, v in data_config_dict.items() if isinstance(v, (int, float, str, bool))}
    data_config_dict = dict(sorted(data_config_dict.items(), key=sort_key))
    data_config_dict = {'data_config':data_config_dict}
    
    merged_config_dict = {**data_config_dict, **ml_params_dict}
    
    trainer_name = 'causal_learning'
    
    wandb.init(
        project='causal-learning',
        name= f'{trainer_name}-{data_config.dataset_name} : {formatted_date}',
        save_code = True,
        monitor_gym = False, 
        config= merged_config_dict
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

def wandb_log_train_metrics(time_cost, lr, ccnet_metric=None, encoder_metric=None, images=None):
    # Define step_logs as a dictionary with relevant key-value pairs
    
    if ccnet_metric is not None:
        gpt_ccnet_losses = dict(ccnet_metric.losses.data)
        gpt_ccnet_errors = dict(ccnet_metric.errors.data)
        
        ccnet_metric = {
        'Core/losses': gpt_ccnet_losses,
        'Core/errors': gpt_ccnet_errors,
        }
    else:
        ccnet_metric = {
        }
        
    if encoder_metric is not None:
        encoder_ccnet_losses = dict(encoder_metric.losses.data)
        encoder_ccnet_errors = dict(encoder_metric.errors.data)
        
        encoder_metric = {
        'Encoder/losses': encoder_ccnet_losses,
        'Encoder/errors': encoder_ccnet_errors,
        }    
    else:
        encoder_metric = {
        }
    
    additional_logs = {"Learning Rate": lr, 
                       "Time Cost": time_cost
                       }
    if images is not None:
        additional_logs["WB Images"] = images
    
    log_data = {**additional_logs}
    train_metrics = {**ccnet_metric, **encoder_metric}
    _wandb_log_data(train_metrics, log_data)

def wandb_log_eval_data(metrics, images):
    additional_logs = {}
    if images is not None:
        additional_logs["WB Images"] = images
        
    log_data = {**additional_logs}
    eval_metrics = {"Evaluate": metrics}
    _wandb_log_data(eval_metrics, log_data=log_data)

