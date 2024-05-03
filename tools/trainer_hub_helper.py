import time
import wandb
import numpy as np
from tools.wandb_logger import wandb_init 
from tools.loader import save_model
from tools.logger import log_train_data, log_test_results
from tools.print import print_iter, print_lr, print_trainer
from nn.utils.init import setup_directories
from tools.wandb_logger import wandb_log_train_data
from tools.display import ImageDebugger
from tools.logger import get_log_name
import logging
import torch
from tools.tensor import adjust_tensor_dim, generate_padding_mask, encode_inputs
from tools.metrics_tracker import MetricsTracker

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
        
        self.use_image = len(self.data_config.obs_shape) != 1
        
        self.gpt_ccnet = self.parent.gpt_ccnet
        self.gpt_trainer = self.parent.gpt_trainer
        
        self.encoder_ccnet = self.parent.encoder_ccnet
        self.encoder_trainer = self.parent.encoder_trainer
        
        self.num_checkpoints = DEFAULT_PRINT_INTERVAL
        self.save_interval = DEFAULT_SAVE_INTERVAL
        
        self.model_path, self.temp_path, self.log_path = setup_directories()
        
        self.logger = logging.getLogger(__name__)
        self.pivot_time = None
        
        self.gpt_metrics = MetricsTracker()
        self.encoder_metrics = MetricsTracker()
        
    def initialize_train(self, dataset):
        # self.sum_losses, self.sum_errors = None, None
        self.iters, self.cnt_checkpoints, self.cnt_print = 0, 0, 0
        if self.use_image:
            self.image_debugger = ImageDebugger(self.gpt_ccnet, dataset, self.data_config, self.device)
    
    def should_checkpoint(self):
        return self.cnt_checkpoints % self.num_checkpoints == 0 and self.cnt_checkpoints != 0
    
    def init_time_step(self):
        if self.pivot_time is None:
            self.pivot_time = time.time()
        
    def save_models(self):
        model_path = self.determine_save_path()
        
        # Lists of components to be saved for GPT
        gpt_network_names = self.gpt_ccnet.network_names
        gpt_networks = self.gpt_ccnet.networks
        gpt_optimizers = self.gpt_trainer.optimizers
        gpt_schedulers = self.gpt_trainer.schedulers
        
        # Lists of components to be saved for encoder
        encoder_network_names = self.encoder_ccnet.network_names
        encoder_networks = self.encoder_ccnet.networks
        encoder_optimizers = self.encoder_trainer.optimizers
        encoder_schedulers = self.encoder_trainer.schedulers
        
        # Ensure all lists are synchronized in length
        assert len(gpt_network_names) == len(gpt_networks) == len(gpt_optimizers) == len(gpt_schedulers), "GPT component lists must be of the same length"
        assert len(encoder_network_names) == len(encoder_networks) == len(encoder_optimizers) == len(encoder_schedulers), "Encoder component lists must be of the same length"
        
        # Iterate over each GPT component set and save
        for model_name, network, optimizer, scheduler in zip(gpt_network_names, gpt_networks, gpt_optimizers, gpt_schedulers):
            save_model(model_path, model_name, network, optimizer, scheduler)
        
        # Iterate over each encoder component set and save
        for model_name, network, optimizer, scheduler in zip(encoder_network_names, encoder_networks, encoder_optimizers, encoder_schedulers):
            save_model(model_path, model_name, network, optimizer, scheduler)
            
    def determine_save_path(self):
        """Determine the file path for saving models based on the current count."""
        return self.model_path if self.cnt_print % 2 == 0 else self.temp_path
            
    def setup_training_step(self, source_batch, target_batch):

        self.init_time_step()

        # Encode inputs to prepare them for causal training
        source_code, target_code = encode_inputs(self.encoder_ccnet, source_batch, target_batch)
        
        # Adjust tensor dimensions for causal processing
        state_trajectory = adjust_tensor_dim(source_code, target_dim=3)  # off when it's img data set
        target_trajectory = adjust_tensor_dim(target_code, target_dim=3)  # off when it's img data set
        
        # Generate padding mask based on state trajectory
        padding_mask = generate_padding_mask(state_trajectory)
        
        return state_trajectory, target_trajectory, padding_mask

    def finalize_training_step(self, epoch_idx, iter_idx, len_dataloader, encoder_metric, gpt_metric, eval_dataset) -> None:
        self.update_metrics(encoder_metric, gpt_metric)

        if self.should_checkpoint():
            self.handle_checkpoint(epoch_idx, iter_idx, len_dataloader, eval_dataset)

        self.increment_counters()

    def update_metrics(self, encoder_metric, gpt_metric):
        """Updates metrics and records time spent since the last checkpoint."""
        self.gpt_metrics += gpt_metric
        self.encoder_metrics += encoder_metric

    def handle_checkpoint(self, epoch_idx, iter_idx, len_dataloader, eval_dataset):
        time_cost = time.time() - self.pivot_time
        wb_image = None

        if self.use_image:
            """Update and log images if using image data."""
            self.image_debugger.update_images()
            image_display = self.image_debugger.display_image()
            if self.use_wandb:
                wb_image = wandb.Image(image_display)

        self.log_checkpoint_details(time_cost, epoch_idx, iter_idx, len_dataloader, wb_image)
        self.save_models()
        self.reset_metrics()

        """Handles operations to be performed at each checkpoint."""
        test_results = self.parent.evaluate(eval_dataset)
        log_test_results(self.tensorboard, self.iters, test_results)

    def log_checkpoint_details(self, time_cost, epoch_idx, iter_idx, len_dataloader, wb_image):
        """Calculates average metrics over the checkpoints."""
        avg_encoder_metric = self.encoder_metrics / float(self.num_checkpoints)
        avg_gpt_metric = self.gpt_metrics / float(self.num_checkpoints)
        
        if self.use_print:
            self.print_checkpoint_info(time_cost, epoch_idx, iter_idx, len_dataloader, avg_encoder_metric, avg_gpt_metric)

        log_train_data(self.tensorboard, self.iters, avg_gpt_metric)
        """Logs training data to Weights & Biases if enabled."""
        if self.use_wandb:
            lr = self.gpt_trainer.get_lr() 
            wandb_log_train_data(avg_encoder_metric, avg_gpt_metric, time_cost=time_cost, lr=lr, images=wb_image)

    def print_checkpoint_info(self, time_cost, epoch_idx, iter_idx, len_dataloader, avg_encoder_metric, avg_gpt_metric):
        """Prints formatted information about the current checkpoint."""
        print_iter(epoch_idx, self.parent.max_epoch, iter_idx, len_dataloader, time_cost)
        print_lr(self.encoder_trainer.optimizers, self.gpt_trainer.optimizers)
        print('--------------------Training Metrics--------------------')
        print_trainer('encoder_ccnet', avg_encoder_metric)
        print_trainer("gpt_ccnet", avg_gpt_metric)

    def reset_metrics(self):
        """resets metrics trackers."""
        self.pivot_time = None
        self.gpt_metrics.reset()
        self.encoder_metrics.reset()
        self.cnt_checkpoints = 0
        self.cnt_print += 1

    def increment_counters(self):
        """Increments general counters for iterations and checkpoints."""
        self.iters += 1
        self.cnt_checkpoints += 1

