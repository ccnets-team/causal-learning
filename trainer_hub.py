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

from tools.loader import get_Dataloader, get_testloader
from nn.utils.init import set_random_seed, setup_directories
from tools.wandb_logger import wandb_end
from tools.print import print_ml_params
from tools.report import calculate_metrics
from tools.wandb_logger import wandb_log_test_data

from framework.ccnet.cooperative_network import CooperativeNetwork
from framework.ccnet.cooperative_encoding_network import CooperativeEncodingNetwork
import random
    
class TrainerHub:
    def __init__(self, ml_params: MLParameters, data_config: DataConfig, device, use_print=False, use_wandb=False):
        print_ml_params(ml_params)
        
        training_params, model_params, optimization_params = ml_params

        d_model = model_params.encoding_params.d_model
        data_config.initialize_(d_model=d_model)
        
        obs_shape = data_config.obs_shape
        stoch_size, det_size = data_config.stoch_size, data_config.det_size
        self.encoder_ccnet = CooperativeEncodingNetwork(model_params, obs_shape, stoch_size, det_size, device)
        self.encoder_ccnet_trainer = CausalEncodingTrainer(self.encoder_ccnet, training_params, optimization_params)

        state_size, label_size, explain_size = data_config.state_size, data_config.label_size, data_config.explain_size
        self.gpt_ccnet = CooperativeNetwork(model_params, state_size, label_size, explain_size, device, encoder=self.encoder_ccnet)
        self.gpt_ccnet_trainer = CausalTrainer(self.gpt_ccnet, training_params, optimization_params)
        
        self.model_path, self.temp_path, self.log_path = setup_directories()
        self.batch_size = ml_params.training.batch_size
        self.num_epoch = ml_params.training.num_epoch

        self.device = device
        self.use_wandb = use_wandb
        self.task_type = data_config.task_type

        self.helper = TrainerHubHelper(self, data_config, ml_params, device, use_print, self.use_wandb)

    def __exit__(self):
        if self.use_wandb:
            wandb_end()
        
    # Main Public Methods
    def train(self, trainset:Dataset, testset:Dataset = None) -> None:
        """
        Train the model based on the provided policy.
        """
        len_trainset = len(trainset)
        batch_size = min(len_trainset, self.batch_size)
        self.helper.initialize_train(trainset)
        
        for epoch in tqdm_notebook(range(self.num_epoch), desc='Epochs', leave=False):
            dataloader = get_Dataloader(trainset, batch_size, shuffle=True)
            len_dataloader = len(dataloader)
            
            # Training loop over each batch 
            for iters, (source_batch, target_batch) in enumerate(tqdm_notebook(dataloader, desc='Iterations', leave=False)):

                set_random_seed(iters)

                # Initial transformations and training preparations
                source_batch, target_batch = self.helper.convert_to_device(source_batch, target_batch)
                
                # Train state encoding model and capture metrics
                obs_encoding_metrics = self.encoder_ccnet_trainer.train_models(source_batch)
                
                state_trajectory, target_trajectory, padding_mask = self.helper.setup_training_step(source_batch, target_batch)

                learning_metrics = self.gpt_ccnet_trainer.train_models(state_trajectory, target_trajectory, padding_mask)
                
                self.helper.finalize_training_step(epoch, iters, len_dataloader, learning_metrics, obs_encoding_metrics, testset)

    def test(self, testset):
        """
        Evaluates the model's performance on a provided test dataset and calculates specified metrics.

        :param testset: The test dataset to evaluate the model on.
        """
        set_random_seed(self.helper.iters)  # Example seed setting
        
        # Set the model to evaluation mode
        self.gpt_ccnet_trainer.set_train(train = False)
        # Determine batch size, capped at the size of the test set
        test_batch_size = min(len(testset), self.batch_size)
        # Get a DataLoader for the test dataset

        dataloader = get_testloader(testset, test_batch_size)
        # Convert dataloader to a list
        dataloader_list = list(dataloader)

        # Select a random batch
        random_batch = random.choice(dataloader_list)
        source_batch, target_batch = random_batch

        # Process the selected batch

        source_batch, target_batch = self.helper.convert_to_device(source_batch, target_batch)
        inferred_y = self.gpt_ccnet.infer(source_batch)

        # Calculate metrics based on the processed batch
        metrics = calculate_metrics(inferred_y, target_batch, self.task_type)

        if self.use_wandb:
            wandb_metrics = {'Test': metrics}
            wandb_log_test_data(wandb_metrics)

        return metrics

