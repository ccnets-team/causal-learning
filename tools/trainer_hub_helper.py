import time
from tools.wandb_logger import wandb_init 
from tools.loader import save_trainer
from tools.logger import tensorboard_log_train_metrics, log_test_results
from tools.print import print_iter, print_lr, print_trainer, print_test_results
from tools.wandb_logger import wandb_log_train_metrics, wandb_log_eval_data, wandb_image
from tools.image_debugger import ImageDebugger
from tools.logger import get_log_name
import os
import logging
from tools.metric_tracker import MetricsTracker

from torch.utils.tensorboard import SummaryWriter

from tools.setting.ml_params import MLParameters 
from tools.setting.data_config import DataConfig

class TrainerHubHelper:
    def __init__(self, parent, data_config: DataConfig, ml_params: MLParameters, device, use_print, use_wandb, print_interval):
        self.parent = parent
        self.device = device
        
        self.use_print = use_print
        self.tensorboard = SummaryWriter(log_dir=get_log_name('../logs'))
        
        self.use_wandb = use_wandb
        
        self.use_image = len(data_config.obs_shape) != 1
        
        self.num_checkpoints = print_interval
        self.model_path, self.temp_path, self.log_path = self.setup_directories()
        
        self.logger = logging.getLogger(__name__)
        self.pivot_time = None
        
        self.use_gpt = self.parent.use_gpt
        self.use_ccnet = self.parent.use_ccnet
        self.use_encoder = self.parent.use_encoder
        
        self.ccnet_metrics = MetricsTracker()
        self.encoder_metrics = MetricsTracker() 
        
        self.use_image_debugger = data_config.show_image_indices is not None
        
        if self.use_image_debugger:
            image_ccnet = self.parent.ccnet if self.use_ccnet else self.parent.encoder 
            self.image_debugger = ImageDebugger(image_ccnet, data_config, self.device, use_ccnet = self.use_ccnet)
        
        self.data_config = data_config 
        self.ml_params = ml_params
        
    def initialize_train(self, dataset):
        if hasattr(dataset, 'max_seq_len'):
            self.ml_params.training.max_seq_len = dataset.max_seq_len
        
        if hasattr(dataset, 'min_seq_len'):
            self.ml_params.training.min_seq_len = dataset.min_seq_len
        
        if self.use_wandb:
            wandb_init(self.data_config, self.ml_params)
        
        self.iters, self.cnt_checkpoints, self.cnt_print = 0, 0, 0
        if self.use_image_debugger:
            self.image_debugger.initialize_(dataset)
    
    def should_checkpoint(self):
        return self.cnt_checkpoints % self.num_checkpoints == 0 and self.cnt_checkpoints != 0
    
    def init_time_step(self):
        if self.pivot_time is None:
            self.pivot_time = time.time()

    def setup_directories(self, base_path = '../'):
        set_model_path = os.path.join(base_path, "models")
        set_temp_path = os.path.join(base_path, "models/temp")
        set_log_path = os.path.join(base_path, "logs")

        for path in [set_model_path, set_temp_path, set_log_path]:
            os.makedirs(path, exist_ok=True)

        return set_model_path, set_temp_path, set_log_path
        
    def determine_save_path(self):
        """Determine the file path for saving models based on the current count."""
        return self.model_path if self.cnt_print % 2 == 0 else self.temp_path

    def finalize_training_step(self, epoch_idx, iter_idx, len_dataloader, core_metric=None, encoder_metric=None, test_results=None) -> None:
        """Perform end-of-step operations including metrics update and checkpointing."""
        self.update_metrics(core_metric, encoder_metric)
        
        if self.should_checkpoint():
            self.perform_checkpoint_operations(epoch_idx, iter_idx, len_dataloader, test_results)

        self.increment_counters()

    def update_metrics(self, core_metric = None, encoder_metric = None):
        """Update training metrics if available."""
        if core_metric is not None:
            self.ccnet_metrics += core_metric
        if encoder_metric is not None:
            self.encoder_metrics += encoder_metric

    def perform_checkpoint_operations(self, epoch_idx, iter_idx, len_dataloader, test_results=None):
        """Handle all operations required at checkpoint: logging, saving, and metrics reset."""
        time_cost = time.time() - self.pivot_time
        wb_image = self.update_image()
        total_iters = epoch_idx * len_dataloader + iter_idx 
        
        self.log_checkpoint_details(time_cost, epoch_idx, iter_idx, len_dataloader, wb_image)
        self.save_trainers()
        self.reset_metrics()

        if self.use_ccnet and test_results is not None:
            self.handle_test_results(test_results)
            if self.use_wandb:
                wandb_log_eval_data(test_results, wb_image, iters = total_iters)
                
    def handle_test_results(self, test_results=None):
        """Print and log test results if core is used."""
        print_test_results(test_results)
        log_test_results(self.tensorboard, self.iters, test_results)

    def increment_counters(self):
        """Increment iteration and checkpoint counters."""
        self.iters += 1
        self.cnt_checkpoints += 1

    def update_image(self):
        if self.use_image_debugger:
            # Update and display images when image data is enabled
            self.image_debugger.update_images()
            image_display = self.image_debugger.display_image()
            
            # Log the image to Wandb if enabled
            if self.use_wandb:
                return wandb_image(image_display)
            
        # Return None if images are not used or logging is not enabled
        return None
    
    def save_trainers(self):
        save_path = self.determine_save_path()
        """Saves the current state of the trainers."""
        if self.use_ccnet:
            save_trainer(save_path, self.parent.ccnet_trainer)
        if self.use_encoder:
            save_trainer(save_path, self.parent.encoder_trainer)
            
    def log_checkpoint_details(self, time_cost, epoch_idx, iter_idx, len_dataloader, wb_image):
        ccnet_trainer = self.parent.ccnet_trainer
        """Calculates average metrics over the checkpoints."""
        avg_ccnet_metric = self.ccnet_metrics / float(self.num_checkpoints) if self.use_ccnet else None
        avg_encoder_metric = self.encoder_metrics / float(self.num_checkpoints) if self.use_encoder else None
        total_iters = epoch_idx * len_dataloader + iter_idx 
        
        if self.use_print:
            self.print_checkpoint_info(time_cost, epoch_idx, iter_idx, len_dataloader, avg_ccnet_metric, avg_encoder_metric)

        tensorboard_log_train_metrics(self.tensorboard, self.iters, ccnet_metric=avg_ccnet_metric, encoder_metric=avg_encoder_metric)
        """Logs training data to Weights & Biases if enabled."""
        if self.use_wandb:
            lr = ccnet_trainer.get_lr() 
            wandb_log_train_metrics(time_cost, lr=lr, ccnet_metric=avg_ccnet_metric, encoder_metric=avg_encoder_metric, images=wb_image, iters = total_iters)

    def print_checkpoint_info(self, time_cost, epoch_idx, iter_idx, len_dataloader, avg_ccnet_metric = None, avg_encoder_metric = None):
        """Prints formatted information about the current checkpoint."""
        ccnet = self.parent.ccnet
        encoder = self.parent.encoder
        ccnet_trainer = self.parent.ccnet_trainer
        encoder_trainer = self.parent.encoder_trainer
         
        print_iter(epoch_idx, self.parent.num_epoch, iter_idx, len_dataloader, time_cost)
        if avg_ccnet_metric is not None and avg_encoder_metric is not None:
            print_lr(encoder_trainer.optimizers, ccnet_trainer.optimizers)
        elif avg_ccnet_metric is not None:
            print_lr(ccnet_trainer.optimizers)
        elif avg_encoder_metric is not None:
            print_lr(encoder_trainer.optimizers)
        print('--------------------Training Metrics--------------------')
        if self.use_encoder and avg_encoder_metric is not None:
            encoder_network_type = "Encoder" 
            encoder_network_name = encoder.model_name.capitalize()
            print_trainer(encoder_network_type, encoder_network_name, avg_encoder_metric)
        if self.use_ccnet and avg_ccnet_metric is not None:
            ccnet_type = "CCNet" 
            ccnet_name = ccnet.model_name.capitalize()
            print_trainer(ccnet_type, ccnet_name, avg_ccnet_metric)

    def reset_metrics(self):
        """resets metrics trackers."""
        self.pivot_time = None
        self.ccnet_metrics.reset()
        self.encoder_metrics.reset()
        self.cnt_checkpoints = 0
        self.cnt_print += 1



