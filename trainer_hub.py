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

from tools.loader import get_data_loader, get_test_loader, load_trainer
from nn.utils.init import set_random_seed
from tools.wandb_logger import wandb_end
from tools.report import calculate_test_results

from framework.ccnet.cooperative_network import CooperativeNetwork
from framework.ccnet.cooperative_encoding_network import CooperativeEncodingNetwork
from tools.setting.ml_config import configure_ccnet_network, configure_encoder_network
from tools.tensor_utils import select_last_sequence_elements, manage_batch_dimensions, prepare_batches, get_random_batch
import torch

DEFAULT_PRINT_INTERVAL = 100

class TrainerHub:
    def __init__(self, ml_params: MLParameters, data_config: DataConfig, device, use_print=False, use_wandb=False, print_interval=DEFAULT_PRINT_INTERVAL):
        self.data_config = data_config
        self.device = device
        
        self.initialize_usage_flags(ml_params)
        self.initialize_training_params(ml_params)
        self.initialize_models()
        
        self.task_type = self.data_config.task_type
        self.label_size = self.data_config.label_size
        
        self.setup_models(ml_params)
        
        self.helper = TrainerHubHelper(self, data_config, ml_params, device, use_print, use_wandb, print_interval)

    def initialize_usage_flags(self, ml_params):
        self.use_ccnet = ml_params.ccnet_network != 'none'
        self.use_gpt = ml_params.ccnet_network == 'gpt'
        self.use_encoder = ml_params.encoder_network != 'none'
        
    def initialize_training_params(self, ml_params):
        training_params = ml_params.training
        batch_size = training_params.batch_size
        self.batch_size = batch_size
        self.eval_batch_size = 4 * batch_size
        self.test_batch_size = 10 * batch_size
        self.num_epoch = training_params.num_epoch
        self.max_iters = training_params.max_iters

    def initialize_models(self):
        self.encoder = None
        self.encoder_trainer = None
        self.ccnet = None
        self.ccnet_trainer = None
        
    def __exit__(self):
        if self.use_wandb:
            wandb_end()
    
    def load_trainer(self, ccnet_network = True):
        if ccnet_network:
            load_trainer(self.helper.model_path, self.ccnet_trainer)
        else:
            load_trainer(self.helper.model_path, self.encoder_trainer)
            
    def setup_models(self, ml_params):
        training_params, algorithm_params, model_params, optimization_params = ml_params
        if self.use_encoder:
            model_networks, network_params = configure_encoder_network(self.data_config, model_params.encoder_network, model_params.encoder_config)
            self.encoder = CooperativeEncodingNetwork(model_networks, network_params, algorithm_params, self.device)
            self.encoder_trainer = CausalEncodingTrainer(self.encoder, training_params, algorithm_params, optimization_params, self.task_type)
        
        if self.use_ccnet:
            model_networks, network_params = configure_ccnet_network(self.data_config, model_params.ccnet_network, model_params.ccnet_config)
            self.ccnet = CooperativeNetwork(model_networks, network_params, algorithm_params, self.task_type, self.device, encoder=self.encoder)
            self.ccnet_trainer = CausalTrainer(self.ccnet, training_params, algorithm_params, optimization_params, self.task_type)

    def train_iteration(self, iters, source_batch, target_batch):
        set_random_seed(iters)
        self.helper.init_time_step()
        
        source_batch, target_batch, padding_mask = prepare_batches(source_batch, target_batch, self.label_size, self.task_type, self.device)        
        encoder_metric = self.encoder_trainer.train_models(source_batch, padding_mask) if self.use_encoder else None

        if self.use_ccnet:
            source_batch, target_batch = self.encode_inputs(source_batch, target_batch, padding_mask)
            source_batch, target_batch, padding_mask = manage_batch_dimensions(self.use_gpt, source_batch, target_batch, padding_mask)
            core_metric = self.ccnet_trainer.train_models(source_batch, target_batch, padding_mask)
        else:
            core_metric = None
            
        return core_metric, encoder_metric
            
    def train(self, trainset: Dataset, testset: Dataset = None):
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

    def validate_performance(self, dataloader):
        all_inferred_batches = []
        all_target_batches = []

        for source_batch, target_batch in dataloader:
            source_batch, target_batch, padding_mask = prepare_batches(source_batch, target_batch, self.label_size, self.task_type, self.device)        
            
            inferred_batch = self.ccnet.infer(source_batch, padding_mask)

            if self.use_gpt and padding_mask is not None:
                inferred_batch, target_batch = select_last_sequence_elements(inferred_batch, target_batch, padding_mask)
                
            all_inferred_batches.append(inferred_batch)
            all_target_batches.append(target_batch)

        all_inferred_batches = torch.cat(all_inferred_batches, dim=0)
        all_target_batches = torch.cat(all_target_batches, dim=0)
        
        test_metrics = calculate_test_results(all_inferred_batches, all_target_batches, self.task_type, num_classes=self.label_size)
        return test_metrics

    def evaluate(self, dataset):
        if not self.use_ccnet or dataset is None or not self.helper.should_checkpoint():
            return None
        random_batch = get_random_batch(dataset, min(len(dataset), self.eval_batch_size))
        return self.validate_performance([random_batch])

    def test(self, dataset, batch_size=None):
        if dataset is None:
            return None              
        if batch_size is None:
            batch_size = self.test_batch_size
        dataloader = get_test_loader(dataset, min(len(dataset), batch_size))
        return self.validate_performance(dataloader)
        
    def encode_inputs(self, observation, labels, padding_mask = None):
        with torch.no_grad():
            encoded_obseartion = observation if self.encoder is None else self.encoder.encode(observation, padding_mask = padding_mask)
        return encoded_obseartion, labels
        
    def should_end_training(self, epoch):
        return self.helper.iters > self.max_iters or epoch > self.num_epoch
    