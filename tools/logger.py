'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import os
from datetime import datetime
from typing import Optional, Dict
from tools.report import *

METRICS_CATEGORY_MAP = {
    'losses': 'Losses',
    'errors': 'Errors'
}

def get_log_name(log_path):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"{log_path}/{current_time}"
    
    # Check if the directory exists, if it does, append a suffix to make it unique
    suffix = 0
    while os.path.isdir(log_dir):
        suffix += 1
        log_dir = f"{log_path}/{current_time}_{suffix}"
    
    return log_dir

def log_data(logger, step, scalar_logs=None, metrics=None):
    # Log scalar data
    for name, value in scalar_logs.items():
        if value is not None:
            logger.add_scalar(name, value, step)

    if metrics is not None:
        # Loop over each metrics category and log each metric with a specific prefix
        for category_name, category_metrics in metrics.__dict__.items():
            # Map the category_name to the new desired name
            mapped_category_name = METRICS_CATEGORY_MAP.get(category_name, category_name.title())
            
            for metric_name, metric_value in category_metrics.items():
                if metric_value is not None:
                    components = metric_name.split('_')
                    new_metric_name = components[-2].title() if len(components) > 1 else components[0].title()                         
                    log_name = f"{mapped_category_name}/{new_metric_name}"
                    logger.add_scalar(log_name, metric_value, step)


def prepare_metrics_data(metric_data: Optional[Dict]):
    """
    Prepare structured metrics data for logging.

    Args:
    metric_data : Metrics data from the training process.

    Returns:
    Structured metrics dictionary with separate keys for losses and errors.
    """
    if metric_data is not None:
        losses = dict(metric_data.losses.data)
        errors = dict(metric_data.errors.data)

        return {
            'losses': losses,
            'errors': errors,
        }
    return {}

def tensorboard_log_train_metrics(tensorboard, iteration: int, core_metric: Optional[Dict], encoder_metric: Optional[Dict]):
    """
    Log training metrics for core and encoder types to TensorBoard.

    Args:
    tensorboard : The TensorBoard logger.
    iteration : The current training iteration.
    core_metric : Core trainer specific metric data.
    encoder_metric : Encoder trainer specific metric data.
    """
    # Prepare metric data
    core_metrics = prepare_metrics_data(core_metric)
    encoder_metrics = prepare_metrics_data(encoder_metric)

    # Logging helper function
    def log_metrics(metrics, trainer_type):
        for key, values in metrics.items():
            for metric_name, metric_value in values.items():
                tensorboard.add_scalar(f"Train/{trainer_type}/{metric_name}", metric_value, iteration)

    # Log metrics for core and encoder
    log_metrics(core_metrics, 'Core')
    log_metrics(encoder_metrics, 'Encoder')

    tensorboard.flush()  # Ensure all metrics are written out

def log_test_results(tensorboard, iters, results):
    """
        Log Test metric into tensorboard.

        Parameters
        ---------
            tensorboard : torch.utils.tensorboard.SummaryWriter, optional
                Object of tensorboard SummaryWriter.
                If None, tensorboard logging will be skipped.
            label_types : Iterable obj of str
                Specifies the type of label for the dataset or the metric to be used for evaluation.
            iters : int
                Iteration of training.
            metric : str
                Specifies the type of label or metric used for evaluation.
    """
    if tensorboard is not None:
        for label_type, metric in results.items():
            tensorboard.add_scalar(f"{label_type}", metric, iters)
        tensorboard.flush()