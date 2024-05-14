'''
Author:
        
        PARK, JunHo, junho@ccnets.org
        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

from tqdm.notebook import tqdm_notebook
from framework.ccnet.causal_trainer import CausalTrainer
from framework.ccnet.causal_encoding_trainer import CausalEncodingTrainer

from tools.trainer_hub_helper import TrainerHubHelper
from tools.setting.ml_params import MLParameters
from tools.setting.data_config import DataConfig
from torch.utils.data import Dataset

from tools.loader import get_data_loader, get_test_loader
from nn.utils.init import set_random_seed
from tools.wandb_logger import wandb_end
from tools.report import calculate_test_results
from tools.tensor_utils import convert_to_device, get_random_batch

from framework.ccnet.cooperative_network import CooperativeNetwork
from framework.ccnet.cooperative_encoding_network import CooperativeEncodingNetwork
from tools.setting.ml_config import configure_core_model, configure_encoder_model
from tools.tensor_utils import generate_padding_mask, extract_last_elements_with_mask
import torch

DEFAULT_PRINT_INTERVAL = 50

class TrainerHub:
    def __init__(self, ml_params: MLParameters, data_config: DataConfig, device, use_print=False, use_wandb=False, print_interval=DEFAULT_PRINT_INTERVAL):
        
        self.data_config = data_config
        self.device = device
        
        self.use_core = ml_params.core_model != 'none'
        self.use_gpt = ml_params.core_model == 'gpt'
        self.use_encoder = ml_params.encoder_model != 'none'

        self.use_print = use_print
        self.use_wandb = use_wandb
        training_params = ml_params.training
        self.batch_size = training_params.batch_size
        self.eval_batch_size = 10 * training_params.batch_size
        self.num_epoch = training_params.num_epoch
        self.max_iters = training_params.max_iters

        self.encoder_ccnet = None
        self.encoder_trainer = None
        self.state_size = None
        
        self.core_ccnet = None
        self.core_trainer = None        
        
        self.setup_models(ml_params)
        
        self.helper = TrainerHubHelper(self, data_config, ml_params, device, use_print, use_wandb, print_interval)

    def __exit__(self):
        if self.use_wandb:
            wandb_end()

    def setup_models(self, ml_params):
        training_params, model_params, optimization_params = ml_params
        
        if self.use_encoder:
            self.setup_encoder(model_params, training_params, optimization_params)
        
        if self.use_core:
            self.setup_core_network(model_params, training_params, optimization_params)

    def setup_encoder(self, model_params, training_params, optimization_params):

        model_networks, network_params = configure_encoder_model(self.data_config, model_params.encoder_model, model_params.encoder_config)
        
        self.encoder_ccnet = CooperativeEncodingNetwork(model_networks, network_params, self.device)
        self.encoder_trainer = CausalEncodingTrainer(self.encoder_ccnet, training_params, optimization_params)

    def setup_core_network(self, model_params, training_params, optimization_params):    
        self.label_size = self.data_config.label_size
        self.task_type = self.data_config.task_type
        
        model_networks, network_params = configure_core_model(self.data_config, model_params.core_model, model_params.core_config)
            
        self.core_ccnet = CooperativeNetwork(model_networks, network_params, self.task_type, self.device, encoder=self.encoder_ccnet)
        self.core_trainer = CausalTrainer(self.core_ccnet, training_params, optimization_params)

    def train_iteration(self, iters, source_batch, target_batch):
        set_random_seed(iters)
        self.helper.init_time_step()
        
        # Prepare batches by moving them to the appropriate device.
        source_batch, target_batch = convert_to_device(source_batch, target_batch, device=self.device)
        
        source_batch, target_batch, padding_mask = generate_padding_mask(source_batch, target_batch)
        
        # Train the encoder if enabled and obtain metrics.
        encoder_metric = self.encoder_trainer.train_models(source_batch, padding_mask) if self.use_encoder else None

        # Train the core model if enabled and obtain metrics.
        if self.use_core:
            source_batch, target_batch = self.helper.encode_inputs(source_batch, target_batch, padding_mask)
            
            state_trajectory, target_trajectory, padding_mask = self.helper.prepare_batch_data(source_batch, target_batch, padding_mask)
            
            core_metric = self.core_trainer.train_models(state_trajectory, target_trajectory, padding_mask)
        else:
            core_metric = None
            
        return core_metric, encoder_metric
            
    def train(self, trainset: Dataset, testset: Dataset = None):
        """
        Train the model based on the provided policy.
        """
        self.helper.initialize_train(trainset)
        
        for epoch in tqdm_notebook(range(self.num_epoch), desc='Epochs', leave=False):
            dataloader = get_data_loader(trainset, min(len(trainset), self.batch_size))

            if self.should_end_training(epoch = epoch):
                break
            # show me the max length of the dataset
            for iters, (source_batch, target_batch) in enumerate(tqdm_notebook(dataloader, desc='Iterations', leave=False)):
                core_metric, encoder_metric = self.train_iteration(iters, source_batch, target_batch)

                test_results = self.evaluate(testset)
                    
                self.helper.finalize_training_step(epoch, iters, len(dataloader), core_metric, encoder_metric, test_results)

    def evaluate(self, dataset):
        if not self.use_core or dataset is None or not self.helper.should_checkpoint():
            return None
        random_batch = get_random_batch(dataset, min(len(dataset), self.eval_batch_size))
        return self._test([random_batch])

    def test(self, dataset, batch_size=None):
        if dataset is None:
            return None              
        if batch_size is None:
            batch_size = self.eval_batch_size
        dataloader = get_test_loader(dataset, min(len(dataset), batch_size))
        return self._test(dataloader)
        
    def _test(self, dataloader):
        all_inferred_batches = []
        all_target_batches = []

        for source_batch, target_batch in dataloader:
            source_batch, target_batch = convert_to_device(source_batch, target_batch, self.device)
            
            source_batch, target_batch, padding_mask = self.helper.prepare_batch_data(source_batch, target_batch)
            
            source_batch, target_batch, padding_mask = generate_padding_mask(source_batch, target_batch)
            
            inferred_batch = self.core_ccnet.infer(source_batch, padding_mask)

            if self.use_gpt:
                inferred_batch, target_batch = extract_last_elements_with_mask(inferred_batch, target_batch, padding_mask)
                
            all_inferred_batches.append(inferred_batch)
            all_target_batches.append(target_batch)

        all_inferred_batches = torch.cat(all_inferred_batches, dim=0)
        all_target_batches = torch.cat(all_target_batches, dim=0)
        
        final_results = calculate_test_results(all_inferred_batches, all_target_batches, self.task_type, num_classes=self.label_size)
        return final_results
    
    def should_end_training(self, epoch):
        return self.helper.iters > self.max_iters or epoch > self.num_epoch
    