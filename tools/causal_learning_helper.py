import time
from tools.logging.wandb_logger import wandb_init 
from tools.IO.loader import save_trainer, setup_directories
from tools.logging.logger import tensorboard_log_train_metrics, log_eval_results
from tools.IO.print import print_checkpoint_info, print_results
from tools.logging.wandb_logger import wandb_log_train_metrics, wandb_log_train_data, wandb_log_eval_data, wandb_image
from tools.debug.image.debugger import ImageDebugger
from tools.logging.logger import get_log_name
import logging
from tools.logging.metric_tracker import MetricsTracker
from torch.utils.tensorboard import SummaryWriter

from tools.config.ml_config import MLConfig 
from tools.config.data_config import DataConfig

class CausalLearningHelper:
    def __init__(self, parent, ml_config: MLConfig, data_config: DataConfig, device, use_print, use_wandb, print_interval):
        self.parent = parent
        self.device = device
        
        self.tensorboard = SummaryWriter(log_dir=get_log_name('../logs'))
        self.logger = logging.getLogger(__name__)
        self.pivot_time = None

        self.ccnet_metrics = MetricsTracker()

        self.num_checkpoints = print_interval
        self.model_path, self.temp_path, self.log_path = setup_directories()
        
        self.use_print = use_print
        self.use_wandb = use_wandb
        self.use_image = len(data_config.obs_shape) != 1
        
        self.use_seq_input = self.parent.use_seq_input
        self.use_image_debugger = data_config.show_image_indices is not None
        
        if self.use_image_debugger:
            image_model = self.parent.ccnet
            self.image_debugger = ImageDebugger(image_model, data_config, self.device)
        
        self.data_config = data_config 
        self.ml_config = ml_config
        self.begin_training = False
        
    def begin_train(self, dataset):
        if self.begin_training:
           return 
        
        self.begin_training = True
        if hasattr(dataset, 'max_seq_len'):
            self.ml_config.training.max_seq_len = dataset.max_seq_len
        
        if hasattr(dataset, 'min_seq_len'):
            self.ml_config.training.min_seq_len = dataset.min_seq_len
        
        if self.use_wandb:
            wandb_init(self.data_config, self.ml_config)
        
        self.iters, self.cnt_checkpoints, self.cnt_print = 0, 0, 0  
        if self.use_image_debugger:
            self.image_debugger.initialize_(dataset)
    
    def should_checkpoint(self):
        return self.cnt_checkpoints % self.num_checkpoints == 0 and self.cnt_checkpoints != 0
    
    def init_time_step(self):
        if self.pivot_time is None:
            self.pivot_time = time.time()

    def determine_save_path(self):
        """Determine the file path for saving models based on the current count."""
        return self.model_path if self.cnt_print % 2 == 0 else self.temp_path

    def finalize_training_step(self, epoch_idx, iter_idx, len_dataloader, 
                               ccnet_metric=None, train_results = None, test_results=None) -> None:
        """Perform end-of-step operations including metrics update and checkpointing."""

        """Update training metrics if available."""
        if ccnet_metric is not None:
            self.ccnet_metrics += ccnet_metric
        
        if self.should_checkpoint():
            self.perform_checkpoint_operations(epoch_idx, iter_idx, len_dataloader, train_results, test_results)

        """Increment iteration and checkpoint counters."""
        self.iters += 1
        self.cnt_checkpoints += 1

    def perform_checkpoint_operations(self, epoch_idx, iter_idx, len_dataloader, train_results=None, test_results=None):
        """Handle all operations required at checkpoint: logging, saving, and metrics reset."""
        time_cost = time.time() - self.pivot_time
        wb_image = self.update_image()
        
        self.log_checkpoint_details(time_cost, epoch_idx, iter_idx, len_dataloader, train_results, wb_image)
        self.save_trainers()
        self.reset_metrics()

        if train_results is not None and self.use_wandb:
            wandb_log_train_data(train_results, wb_image, iters = self.iters)
                
        if test_results is not None:
            self.handle_results(test_results)
            if self.use_wandb:
                wandb_log_eval_data(test_results, wb_image, iters = self.iters)
                
    def handle_results(self, test_results=None):
        """Print and log test results if core is used."""
        if self.use_print:
            print_results(test_results, is_eval=True)
        log_eval_results(self.tensorboard, self.iters, test_results)

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
        trainer = self.parent.trainer
    
        save_trainer(save_path, trainer)
            
    def log_checkpoint_details(self, time_cost, epoch_idx, iter_idx, len_dataloader, train_results, wb_image):
        trainer = self.parent.trainer
        
        """Calculates average metrics over the checkpoints."""
        avg_ccnet_metric = self.ccnet_metrics / float(self.num_checkpoints) 
        
        if self.use_print:
            print_checkpoint_info(self.parent, time_cost, epoch_idx, iter_idx, len_dataloader, avg_ccnet_metric, train_results)

        tensorboard_log_train_metrics(self.tensorboard, self.iters, ccnet_metric=avg_ccnet_metric)
        """Logs training data to Weights & Biases if enabled."""
        if self.use_wandb:
            lr = trainer.get_lr() 
            wandb_log_train_metrics(time_cost, lr=lr, ccnet_metric=avg_ccnet_metric, images=wb_image, iters = self.iters)

    def reset_metrics(self):
        """resets metrics trackers."""
        self.pivot_time = None
        self.ccnet_metrics.reset()
        self.cnt_checkpoints = 0
        self.cnt_print += 1