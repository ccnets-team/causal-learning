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

def print_trainer(trainer_name, losses, errors):
    print('--------------------Training Metrics--------------------')
    print("Trainer: ", trainer_name)
    print('Inf: %.4f\tGen: %.4f\tRec: %.4f\tE: %.4f\tR: %.4f\tP: %.4f'
        % (losses['inference_loss'], losses['generation_loss'], losses['reconstruction_loss'], errors['explainer_error'], errors['reasoner_error'], errors['producer_error']))

def print_lr(optimizers):
    cur_lr_E = optimizers[0].param_groups[0]['lr']
    cur_lr_R = optimizers[1].param_groups[0]['lr']
    cur_lr_P = optimizers[2].param_groups[0]['lr']
    name_opt_E = type(optimizers[0]).__name__
    name_opt_R = type(optimizers[1]).__name__
    name_opt_P = type(optimizers[2]).__name__
    
    if cur_lr_E == cur_lr_R == cur_lr_P and name_opt_E == name_opt_R == name_opt_P:
        print('Opt-{0} lr_ERP: {1}'.format(type(optimizers[0]).__name__, optimizers[0].param_groups[0]['lr']))
    else:
        print('Opt-{0} lr_ERP: {1}'.format(type(optimizers[0]).__name__, optimizers[0].param_groups[0]['lr']))
        print('Opt-{0} lr_ERP: {1}'.format(type(optimizers[1]).__name__, optimizers[1].param_groups[0]['lr']))
        print('Opt-{0} lr_ERP: {1}'.format(type(optimizers[2]).__name__, optimizers[2].param_groups[0]['lr']))

def print_metrics(metrics):
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
        
        
def print_ml_params(cl_params):
    for config_category in ['training', 'model', 'optimization']:
        config = getattr (cl_params, config_category)
        print(f"--- {config_category.upper()} CONFIGURATION ---")

        for attr_name in vars(config):  
            attr_value = getattr(config, attr_name)
            if isinstance(attr_value, torch.device):
                print(f"{attr_name}: {attr_value.type} - {attr_value.index}")
            elif isinstance(attr_value, torch.nn.modules.sparse.Embedding):
                print(f"{attr_name}: Embedding({attr_value.num_embeddings}, {attr_value.embedding_dim})")
            else:
                print(f"{attr_name}: {attr_value}")
        print()
        
        
def print_vae_metrics(vae_metrics):
    print('-------------------- VAE Metrics -------------------------')
    for key, value in vae_metrics.items():
        print(f"{key}: {value:.4f}")
