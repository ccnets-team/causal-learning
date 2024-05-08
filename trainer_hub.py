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
from tools.wandb_logger import log_to_wandb
from tools.tensor_utils import get_random_batch, convert_to_device 

from framework.ccnet.cooperative_network import CooperativeNetwork
from framework.ccnet.cooperative_encoding_network import CooperativeEncodingNetwork
from tools.setting.ml_config import configure_core_model, configure_encoder_model
from tools.tensor_utils import generate_padding_mask

class TrainerHub:
    def __init__(self, ml_params: MLParameters, data_config: DataConfig, device, use_print=False, use_wandb=False, use_full_eval = False):
        
        self.data_config = data_config
        self.device = device
        
        self.use_core = ml_params.core_model_name != 'none'
        self.use_gpt = ml_params.core_model_name == 'gpt'
        self.use_encoder = ml_params.encoder_model_name != 'none'

        self.use_print = use_print
        self.use_wandb = use_wandb
        self.use_full_eval = use_full_eval
        training_params = ml_params.training
        self.batch_size = training_params.batch_size
        self.max_epoch = training_params.max_epoch
        self.max_iters = training_params.max_iters

        self.encoder_ccnet = None
        self.encoder_trainer = None
        self.state_size = None
        
        self.core_ccnet = None
        self.core_trainer = None        
        
        self.setup_models(ml_params)
        
        self.helper = TrainerHubHelper(self, data_config, ml_params, device, use_print, use_wandb)

    def setup_models(self, ml_params):
        training_params, model_params, optimization_params = ml_params
        
        if self.use_encoder:
            self.setup_encoder(model_params, training_params, optimization_params)
        
        if self.use_core:
            self.setup_core_network(model_params, training_params, optimization_params)

    def setup_encoder(self, model_params, training_params, optimization_params):

        model_networks, network_params = configure_encoder_model(self.data_config, model_params.encoder_model_name, model_params.encoding_params)
        
        self.encoder_ccnet = CooperativeEncodingNetwork(model_networks, network_params, self.device)
        self.encoder_trainer = CausalEncodingTrainer(self.encoder_ccnet, training_params, optimization_params)
        

    def setup_core_network(self, model_params, training_params, optimization_params):    
        self.label_size = self.data_config.label_size
        self.task_type = self.data_config.task_type
        
        model_networks, network_params = configure_core_model(self.data_config, model_params.core_model_name, model_params.core_params)
            
        self.core_ccnet = CooperativeNetwork(model_networks, network_params, self.task_type, self.device, encoder=self.encoder_ccnet)
        self.core_trainer = CausalTrainer(self.core_ccnet, training_params, optimization_params)

    def __exit__(self):
        if self.use_wandb:
            wandb_end()

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
            state_trajectory, target_trajectory, padding_mask = self.helper.setup_training_data(source_batch, target_batch, padding_mask)
            core_metric = self.core_trainer.train_models(state_trajectory, target_trajectory, padding_mask)
        else:
            core_metric = None
            
        return core_metric, encoder_metric
            
    def train(self, trainset: Dataset, testset: Dataset = None):
        """
        Train the model based on the provided policy.
        """
        self.helper.initialize_train(trainset)
        
        for epoch in tqdm_notebook(range(self.max_epoch), desc='Epochs', leave=False):
            dataloader = get_data_loader(trainset, min(len(trainset), self.batch_size))

            if self.should_end_training(epoch = epoch):
                break

            for iters, (source_batch, target_batch) in enumerate(tqdm_notebook(dataloader, desc='Iterations', leave=False)):
                core_metric, encoder_metric = self.train_iteration(iters, source_batch, target_batch)

                test_results = self.evaluate(testset)
                    
                self.helper.finalize_training_step(epoch, iters, len(dataloader), core_metric, encoder_metric, test_results)

    def evaluate(self, eval_dataset):
        if not self.use_core or not self.helper.should_checkpoint():
            return None
        source_batch, target_batch = eval_dataset[:] if self.use_full_eval else get_random_batch(eval_dataset, self.batch_size)
        
        # Assuming convert_to_device is a function that handles device placement
        source_batch, target_batch = convert_to_device(source_batch, target_batch, self.device)
        
        source_batch, target_batch, padding_mask = generate_padding_mask(source_batch, target_batch)
        
        inferred_trajectory = self.core_ccnet.infer(source_batch, padding_mask)
        
        test_results = calculate_test_results(inferred_trajectory, target_batch, padding_mask, self.task_type, num_classes=self.label_size)
        
        return test_results

    def should_end_training(self, epoch):
        return self.helper.iters > self.max_iters or epoch > self.max_epoch
