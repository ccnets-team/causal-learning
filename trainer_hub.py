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

from tools.loader import get_data_loader, get_test_loader, _load_trainer
from nn.utils.init import set_random_seed
from tools.wandb_logger import wandb_end
from tools.report import calculate_test_results
from tools.print import print_ml_params, DEFAULT_PRINT_INTERVAL

from framework.ccnet.cooperative_network import CooperativeNetwork
from framework.ccnet.cooperative_encoding_network import CooperativeEncodingNetwork
from tools.setting.ml_config import configure_ccnet_network, configure_encoder_network, _determine_max_iters_and_epoch
from tools.tensor_utils import select_last_sequence_elements, manage_batch_dimensions, prepare_batches, get_random_batch
import torch

class TrainerHub:
    def __init__(self, ml_params: MLParameters, data_config: DataConfig, device, use_print=False, use_wandb=False, print_interval=DEFAULT_PRINT_INTERVAL):
        self.data_config = data_config
        self.device = device
        
        
        self.initialize_usage_flags(ml_params)
        
        self.task_type = self.data_config.task_type
        self.label_size = self.data_config.label_size
        
        self.setup_models(ml_params)
        
        print_ml_params("causal_trainer", ml_params, data_config)
        
        self.helper = TrainerHubHelper(self, data_config, ml_params, device, use_print, use_wandb, print_interval)
        
    def __exit__(self):
        if self.helper.use_wandb:
            wandb_end()

    def initialize_usage_flags(self, ml_params):
        self.use_encoder = ml_params.encoder_network != 'none'
        self.use_ccnet = ml_params.ccnet_network != 'none'
        self.use_seq = ml_params.ccnet_network == 'gpt'
        
    def initialize_training_params(self, ml_params):
        
        _determine_max_iters_and_epoch(ml_params)
        
        training_params = ml_params.training
        batch_size = training_params.batch_size
        self.batch_size = batch_size
        self.eval_batch_size = 4 * batch_size
        self.test_batch_size = 10 * batch_size
        self.num_epoch = training_params.num_epoch
        self.max_iters = training_params.max_iters
    
    def load_trainer(self, ccnet_network = True):
        if ccnet_network:
            _load_trainer(self.helper.model_path, self.ccnet_trainer)
        else:
            _load_trainer(self.helper.model_path, self.encoder_trainer)
            
    def setup_models(self, ml_params):
        model_params, training_params, optimization_params, algorithm_params = ml_params
        if self.use_encoder:
            encoder_networks, network_params = configure_encoder_network(model_params.encoder_network, model_params.encoder_config, self.data_config)
            self.encoder = CooperativeEncodingNetwork(encoder_networks, network_params, algorithm_params, self.device)
            self.encoder_trainer = CausalEncodingTrainer(self.encoder, algorithm_params, optimization_params, self.data_config)
        else:
            self.encoder = None
            self.encoder_trainer = None
        
        if self.use_ccnet:
            ccnet_networks, network_params = configure_ccnet_network(model_params.ccnet_network, model_params.ccnet_config, self.data_config)
            self.ccnet = CooperativeNetwork(ccnet_networks, network_params, algorithm_params, self.data_config, self.device, encoder=self.encoder)
            self.ccnet_trainer = CausalTrainer(self.ccnet, algorithm_params, optimization_params, self.data_config)
        else:
            self.ccnet = None
            self.ccnet_trainer = None
        
    def train_iteration(self, iters, source_batch, target_batch):
        self.start_iteration(iters)
        
        source_batch, target_batch, padding_mask = prepare_batches(source_batch, target_batch, self.label_size, self.task_type, self.device)        
        if self.use_encoder:
            encoder_metric = self.encoder_trainer.train_models(source_batch, padding_mask)
        else:
            encoder_metric = None

        if self.use_ccnet:
            source_batch, target_batch = self.encode_inputs(source_batch, target_batch, padding_mask)
            source_batch, target_batch, padding_mask = manage_batch_dimensions(self.use_seq, source_batch, target_batch, padding_mask)
            ccnet_metric = self.ccnet_trainer.train_models(source_batch, target_batch, padding_mask)
        else:
            ccnet_metric = None
            
        return ccnet_metric, encoder_metric
            
    def train(self, trainset: Dataset, testset: Dataset = None):
        self.helper.begin_train(trainset)
        
        for epoch in tqdm_notebook(range(self.num_epoch), desc='Epochs', leave=False):
            dataloader = get_data_loader(trainset, min(len(trainset), self.batch_size))

            # show me the max length of the dataset
            for iters, (source_batch, target_batch) in enumerate(tqdm_notebook(dataloader, desc='Iterations', leave=False)):
                ccnet_metric, encoder_metric = self.train_iteration(iters, source_batch, target_batch)

                test_results = self.evaluate(testset)
                    
                self.helper.finalize_training_step(epoch, iters, len(dataloader), ccnet_metric, encoder_metric, test_results)

    def evaluate(self, dataset):
        if dataset is None or not self.use_ccnet:
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
        return self.validate_performance(dataloader)

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
        
        test_metrics = calculate_test_results(all_inferred_batches, all_target_batches, self.task_type, num_classes=self.label_size)
        return test_metrics
    
    def should_select_last_sequence(self, padding_mask):
        return self.use_seq and padding_mask is not None
        
    def start_iteration(self, iters):
        set_random_seed(iters)
        self.helper.init_time_step()
        
    def encode_inputs(self, observation, labels, padding_mask = None):
        with torch.no_grad():
            encoded_observation = observation if self.encoder is None else self.encoder.encode(observation, padding_mask = padding_mask)
        return encoded_observation, labels
        