import time
import wandb
import numpy as np
from tools.wandb_logger import wandb_init 
from tools.loader import save_model
from tools.logger import log_train_data, log_test_data
from tools.print import print_iter, print_lr, print_trainer
from nn.utils.init import setup_directories
from tools.wandb_logger import wandb_log_train_data
from tools.display import ImageDebugger
from tools.logger import get_log_name
import logging
import torch
from tools.tensor import adjust_tensor_dim

from torch.utils.tensorboard import SummaryWriter

DEFAULT_PRINT_INTERVAL = 50
DEFAULT_SAVE_INTERVAL = 1000

class TrainerHubHelper:
    def __init__(self, parent, data_config, ml_params, device, use_print, use_wandb):
        self.parent = parent
        self.device = device
        
        self.use_print = use_print
        self.tensorboard = SummaryWriter(log_dir=get_log_name('./logs'))
        self.data_config = data_config
        if use_wandb:
            wandb_init(data_config, ml_params)
        self.use_wandb = use_wandb
        self.num_epoch = ml_params.training.num_epoch
        self.obs_shape = self.data_config.obs_shape
        self.use_image = len(self.obs_shape) != 1
        self.label_size, self.selected_indices = data_config.label_size, ml_params.selected_indices
        
        self.num_checkpoints = DEFAULT_PRINT_INTERVAL
        self.save_interval = DEFAULT_SAVE_INTERVAL
        self.logger = logging.getLogger(__name__)

        self.model_path, self.temp_path, self.log_path = setup_directories()
        
    def initialize_train(self, dataset):
        self.sum_losses, self.sum_errors = None, None
        self.iters, self.cnt_checkpoints, self.cnt_print = 0, 0, 0
        self.pvt_time = time.time()
        if self.use_image:
            self.image_debugger = ImageDebugger(self.parent.gpt_ccnet, dataset, self.data_config, self.device, self.selected_indices)
    
    def should_checkpoint(self):
        return self.cnt_checkpoints % self.num_checkpoints == 0 and self.cnt_checkpoints != 0
    
    def _time_checkpoint(self):
        cur_time = time.time()
        et = cur_time - self.pvt_time
        self.pvt_time = cur_time
        return et
    
    def _save_models(self, model_path = None):
        ccnet = self.parent.gpt_ccnet
        ccnet_trainer = self.parent.gpt_ccnet_trainer
        
        model_path = self.model_path if model_path is None else model_path

        for model_name, network, opimizer, scheduler in zip(ccnet.network_names, ccnet.networks, ccnet_trainer.optimizers, ccnet_trainer.schedulers):
            save_model(model_path, model_name, network, opimizer, scheduler)
            
    def process_checkpoint(self, epoch_idx, iter_idx, len_dataloader, gpt_ccnet_metrics, encoding_ccnet_metrics, testset):
        wb_image = None
         # If the data type is image, update and display images using the image debugger, then log them with wandb
        if self.use_image:
            self.image_debugger.update_images()
            tmp_image_array = self.image_debugger.display_image()
            wb_image = wandb.Image(tmp_image_array)
        
        # Record the elapsed time for this checkpoint
        et = self._time_checkpoint()
        
        optimizers = self.parent.gpt_ccnet_trainer.optimizers
        
        if self.use_print:
            print_iter(epoch_idx, self.num_epoch, iter_idx, len_dataloader, et)
            print_lr(optimizers)
            print_trainer("encoder_ccnet", encoding_ccnet_metrics.losses, encoding_ccnet_metrics.errors)
            print_trainer("gpt_ccnet", gpt_ccnet_metrics.losses, gpt_ccnet_metrics.errors)
      
         # Log training data to TensorBoard if enabled
        log_train_data(self.tensorboard, self.iters, gpt_ccnet_metrics.losses, gpt_ccnet_metrics.errors)
         # Determine the model saving path based on the iteration count and save the models
        save_path = self.model_path if self.cnt_print %2 == 0 else self.temp_path
        self._save_models(model_path = save_path)

        # If label types are specified and a testset is provided, evaluate the model and log the metrics
        if testset != None:
            metrics = self.parent.test(testset) 
            log_test_data(self.tensorboard, self.iters, metrics = metrics)
        
        self.sum_losses, self.sum_errors, self.parent.cnt_checkpoints = None, None, 0
        self.cnt_print += 1
        
        return metrics, wb_image
    
    def record_training_results(self, metrics):
        losses = metrics.losses
        errors = metrics.errors

        losses_mean = np.mean([losses['inference_loss'], losses['generation_loss'], losses['reconstruction_loss']])
        errors_mean = np.mean([errors['explainer_error'], errors['reasoner_error'], errors['producer_error']])
        self.sum_losses = (self.sum_losses + losses_mean) if self.sum_losses is not None else losses_mean
        self.sum_errors = (self.sum_errors + errors_mean) if self.sum_errors is not None else errors_mean
        return losses_mean, errors_mean
        
    def generate_padding_mask(self, source_batch):
        """
        Generate a padding mask for the source batch where all -inf values are masked.
        Args:
        source_batch (torch.Tensor): Tensor of shape [batch_size, seq_len, obs_size]
        Returns:
        torch.Tensor: A 3D tensor of shape [batch_size, seq_len, 1] where padded elements are 1.
        """
        # Identify positions that are -inf (assuming -inf represents padding)
        padding_mask = (source_batch == float('-inf')).any(dim=-1)

        padding_mask = ~padding_mask
        
        return padding_mask.unsqueeze(-1).float()

    def convert_to_device(self, source_batch, target_batch):
        source_batch, target_batch = source_batch.float().to(self.device), target_batch.float().to(self.device)
        return source_batch, target_batch
         
        # Train models and return losses and errors

    def setup_training_step(self, source_batch, target_batch):

        # Encode inputs to prepare them for causal training
        source_code, target_code = self.encode_inputs(source_batch, target_batch)
        
        # Adjust tensor dimensions for causal processing
        source_trajectory = adjust_tensor_dim(source_code, target_dim=3)  # off when it's img data set
        target_trajectory = adjust_tensor_dim(target_code, target_dim=3)  # off when it's img data set
        
        # Generate padding mask based on state trajectory
        padding_mask = self.generate_padding_mask(source_trajectory)
        
        return source_trajectory, target_trajectory, padding_mask
    
    def encode_inputs(self, observation, labels):
        with torch.no_grad():
            encoder = self.parent.encoder_ccnet
            encoded_obseartion = observation if encoder is None else encoder.encode(observation)
        return encoded_obseartion, labels

    def finalize_training_step(self, epoch_idx, iter_idx, len_dataloader, gpt_ccnet_metrics, encoder_ccnet_metrics, testset) -> None:
        wb_image = None
        # Update cumulative losses and errors
        self.record_training_results(gpt_ccnet_metrics)

        # Process checkpoint for logging and saving model if conditions met
        if self.should_checkpoint():
            metrics, wb_image = self.process_checkpoint(epoch_idx, iter_idx, len_dataloader, gpt_ccnet_metrics, encoder_ccnet_metrics, testset)
            
        self.iters += 1; self.cnt_checkpoints += 1
        
        current_lr = self.parent.gpt_ccnet_trainer.get_lr()                
        time_cost = time.time() - self.pvt_time
        # If wandb logging is enabled and there are wandb images to log, log the training data 
        if self.use_wandb:
            if wb_image:
                wandb_log_train_data(gpt_ccnet_metrics, encoder_ccnet_metrics, time_cost=time_cost, lr = current_lr, images=wb_image)
            else:
                wandb_log_train_data(gpt_ccnet_metrics, encoder_ccnet_metrics, time_cost=time_cost, lr = current_lr)