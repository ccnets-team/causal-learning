'''
Author:
        
        PARK, JunHo, junho@ccnets.org
        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

from tqdm.notebook import tqdm_notebook
from ccnet.causal_trainer import CausalTrainer
from ccnet.causal_cooperative_net import CausalCooperativeNet

from tools.causal_learning_helper import CausalLearningHelper
from tools.setting.ml_params import MLParameters
from tools.setting.data_config import DataConfig
from torch.utils.data import Dataset

from tools.loader import get_data_loader, get_test_loader, _load_trainer
from tools.wandb_logger import wandb_end, wandb_log_test_data
from tools.report import calculate_test_results
from tools.print import print_ml_params, DEFAULT_PRINT_INTERVAL

from tools.setting.ml_config import determine_max_iters_and_epoch, update_model_params_from_data, configure_networks
from tools.tensor_utils import select_last_sequence_elements, manage_batch_dimensions, prepare_batches, get_random_batch
from nn.utils.init_layer import set_random_seed
import torch

class CausalLearning:
    def __init__(self, ml_params: MLParameters, data_config: DataConfig, device, use_print=False, use_wandb=False, print_interval=DEFAULT_PRINT_INTERVAL):
        self.data_config = data_config
        self.device = device
        
        update_model_params_from_data(self.data_config, ml_params.model)
        self.initialize_usage_flags(ml_params.model)
        
        self.task_type = self.data_config.task_type
        self.label_size = self.data_config.label_size
        self.label_scale = self.data_config.label_scale
        
        self.setup_models(ml_params)
        self.initialize_training_params(ml_params)
        
        print_ml_params("causal_trainer", ml_params, data_config)
        
        self.helper = CausalLearningHelper(self, data_config, ml_params, device, use_print, use_wandb, print_interval)
        
    def __exit__(self):
        if self.helper.use_wandb:
            wandb_end()

    def initialize_usage_flags(self, model_params):
        self.use_seq_input = model_params.use_seq_input
        
    def initialize_training_params(self, ml_params):
        
        determine_max_iters_and_epoch(ml_params)
        training_params = ml_params.training
        batch_size = training_params.batch_size
        self.batch_size = batch_size
        self.eval_batch_size = 4 * batch_size
        self.test_batch_size = 10 * batch_size
        self.num_epoch = training_params.num_epoch
        self.max_iters = training_params.max_iters
    
    def load_trainer(self):
        _load_trainer(self.helper.model_path, self.trainer)
            
    def setup_models(self, ml_params):
        model_params, training_params, optimization_params = ml_params
        networks = configure_networks(model_params)
        self.ccnet = CausalCooperativeNet(networks, model_params, self.device)
        self.trainer = CausalTrainer(self.ccnet, model_params, training_params, optimization_params)
        
    def train_iteration(self, source_batch, target_batch):
        self.start_iteration()
        
        source_batch, target_batch, padding_mask = prepare_batches(source_batch, target_batch, self.label_size, self.task_type, self.device)        

        source_batch, target_batch, padding_mask = manage_batch_dimensions(self.use_seq_input, source_batch, target_batch, padding_mask)
        ccnet_metric = self.trainer.train_models(source_batch, target_batch, padding_mask)
            
        return ccnet_metric
            
    def train(self, trainset: Dataset, testset: Dataset = None):
        self.helper.begin_train(trainset)
        
        for epoch in tqdm_notebook(range(self.num_epoch), desc='Epochs', leave=False):
            dataloader = get_data_loader(trainset, min(len(trainset), self.batch_size))

            for iters, (source_batch, target_batch) in enumerate(tqdm_notebook(dataloader, desc='Iterations', leave=False)):
                ccnet_metric = self.train_iteration(source_batch, target_batch)

                train_results = self.evaluate(trainset)
                
                test_results = self.evaluate(testset)

                self.helper.finalize_training_step(epoch, iters, len(dataloader), ccnet_metric, train_results, test_results)

    def evaluate(self, dataset):
        if dataset is None:
            return None
        
        if self.helper.should_checkpoint():
            random_batch = get_random_batch(dataset, min(len(dataset), self.eval_batch_size))
            return self.validate_performance([random_batch])
        else:
            return None

    def test(self, dataset, batch_size=None):
        if dataset is None:
            return None              
        if batch_size is None:
            batch_size = self.test_batch_size
        dataloader = get_test_loader(dataset, min(len(dataset), batch_size))
    
        test_metrics = self.validate_performance(dataloader)
        
        wandb_log_test_data(test_metrics, iters=self.helper.iters)

        return test_metrics
        
    def validate_performance(self, dataloader):
        all_inferred_batches = []
        all_target_batches = []

        for source_batch, target_batch in dataloader:
            source_batch, target_batch, padding_mask = prepare_batches(source_batch, target_batch, self.label_size, self.task_type, self.device)        
            
            inferred_batch = self.ccnet.infer(source_batch, padding_mask)

            if self.should_select_last_sequence(padding_mask):
                inferred_batch, target_batch = select_last_sequence_elements(inferred_batch, target_batch, padding_mask)
                
            all_inferred_batches.append(inferred_batch)
            all_target_batches.append(target_batch)

        all_inferred_batches = torch.cat(all_inferred_batches, dim=0)
        all_target_batches = torch.cat(all_target_batches, dim=0)
        
        test_metrics = calculate_test_results(all_inferred_batches, all_target_batches, self.task_type, self.label_size, self.label_scale)
        return test_metrics
    
    def should_select_last_sequence(self, padding_mask):
        return self.use_seq_input and padding_mask is not None
        
    def start_iteration(self):
        set_random_seed(self.helper.iters)
        self.helper.init_time_step()