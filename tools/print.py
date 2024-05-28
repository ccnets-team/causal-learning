'''-ResNet in PyTorch.

Reference:
[1] Modified by PARK, JunHo in April 10, 2022

[2] Writed by Jinsu, Kim in August 11, 2022

[3] Writed by JeongYoong, Kim in April 24, 2024

COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''

import torch

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

def print_test_results(metrics):
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
    print('--------------------Test Metrics------------------------')
    for label_type, metric in metrics.items():
        text = (f'{label_type}: %.4f'
            % metric)
        print(text)
    print()