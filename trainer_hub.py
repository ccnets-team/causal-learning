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
from tools.tensor import convert_to_device
import torch

from framework.ccnet.cooperative_network import CooperativeNetwork
from framework.ccnet.cooperative_encoding_network import CooperativeEncodingNetwork
import random
    
class TrainerHub:
    def __init__(self, ml_params: MLParameters, data_config: DataConfig, device, use_encoder=True, use_print=False, use_wandb=False):
        self.data_config = data_config
        self.device = device
        self.use_encoder = use_encoder
        self.use_print = use_print
        self.use_wandb = use_wandb
        training_params = ml_params.training
        self.batch_size = training_params.batch_size
        self.max_epoch = training_params.max_epoch
        self.max_iters = training_params.max_iters

        self.setup_models(ml_params)
        
        self.helper = TrainerHubHelper(self, data_config, ml_params, device, use_print, use_wandb)

    def setup_models(self, ml_params):
        training_params, model_params, optimization_params = ml_params

        self.setup_encoder(model_params, training_params, optimization_params)

        self.setup_gpt_network(model_params, training_params, optimization_params)

    def setup_encoder(self, model_params, training_params, optimization_params):
        if self.use_encoder:
            obs_shape = self.data_config.obs_shape
            d_model = model_params.encoding_params.d_model
            stoch_size, det_size = d_model, d_model
            self.encoder_ccnet = CooperativeEncodingNetwork(model_params, obs_shape, stoch_size, det_size, self.device)
            self.encoder_trainer = CausalEncodingTrainer(self.encoder_ccnet, training_params, optimization_params)
            self.state_size = stoch_size + det_size
        else:
            self.encoder_ccnet = None
            self.encoder_trainer = None
            self.state_size = self.data_config.obs_shape[-1]

    def setup_gpt_network(self, model_params, training_params, optimization_params):    
        self.task_type = self.data_config.task_type
        self.label_size = self.data_config.label_size
        explain_size = max(model_params.core_params.d_model//2, 1)
        self.gpt_ccnet = CooperativeNetwork(model_params, self.task_type, self.state_size, self.label_size, explain_size, self.device, encoder=self.encoder_ccnet)
        self.gpt_trainer = CausalTrainer(self.gpt_ccnet, training_params, optimization_params)


    def __exit__(self):
        if self.use_wandb:
            wandb_end()
        
    def train(self, trainset: Dataset, testset: Dataset = None):
        """
        Train the model based on the provided policy.
        """
        self.initialize_training(trainset)
        
        for epoch in tqdm_notebook(range(self.max_epoch), desc='Epochs', leave=False):
            dataloader = get_data_loader(trainset, self.adjusted_batch_size(len(trainset)))

            if self.should_end_training(epoch = epoch):
                break

            for iters, (source_batch, target_batch) in enumerate(tqdm_notebook(dataloader, desc='Iterations', leave=False)):
                self.train_iteration(iters, source_batch, target_batch, epoch, len(dataloader), testset)

    def evaluate(self, eval_dataset):
        batch_size = self.batch_size
        num_batches = len(eval_dataset) // batch_size
        
        random_index = random.randint(0, num_batches - 1) if num_batches > 0 else 0
        start_index = random_index * batch_size
        end_index = start_index + batch_size
        
        batch = [eval_dataset[i] for i in range(start_index, min(end_index, len(eval_dataset)))]
        source_batch, target_batch = zip(*batch)
        
        # Convert tuples to tensors if not already tensors
        source_batch = torch.stack(source_batch)  # Assumes source_batch elements are tensors
        if all(isinstance(x, torch.Tensor) for x in target_batch):
            target_batch = torch.stack(target_batch)  # Use torch.stack if target_batch elements are tensors
        else:
            target_batch = torch.tensor(target_batch)  # This line assumes all elements are numeric
        
        source_batch, target_batch = convert_to_device(source_batch, target_batch, self.device)

        inferred_y = self.gpt_ccnet.infer(source_batch)
        test_results = calculate_test_results(inferred_y, target_batch, self.task_type, label_size=self.label_size)
        
        if self.use_wandb:
            log_to_wandb({'Test': test_results})
        
        return test_results

    def should_end_training(self, epoch):
        return self.helper.iters > self.max_iters or epoch > self.max_epoch

    # Helper methods
    def initialize_training(self, trainset):
        self.helper.initialize_train(trainset)
        
    def adjusted_batch_size(self, dataset_length):
        return min(dataset_length, self.batch_size)
        
    def train_iteration(self, iters, source_batch, target_batch, epoch, dataloader_length, testset):
        set_random_seed(iters)
        
        source_batch, target_batch = convert_to_device(source_batch, target_batch, device=self.device)
        
        encoder_metric = self.encoder_trainer.train_models(source_batch) if self.use_encoder else None
        
        state_trajectory, target_trajectory, padding_mask = self.helper.setup_training_step(source_batch, target_batch)
        
        gpt_metric = self.gpt_trainer.train_models(state_trajectory, target_trajectory, padding_mask)
        
        self.helper.finalize_training_step(testset, epoch, iters, dataloader_length, gpt_metric, encoder_metric)
