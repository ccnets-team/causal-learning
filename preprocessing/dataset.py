'''
Author:
        
        PARK, JunHo, junho@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import torch
import pandas as pd
import numpy as np
from nn.utils.init_layer import set_random_seed
 
PRE_BATCH_SIZE = 64

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_X, dataset_y=None, dataset_indices = None):

        self.X = dataset_X.iloc[:].values if isinstance(dataset_X, pd.DataFrame) else dataset_X
        self.y = dataset_y.iloc[:].values if isinstance(dataset_y, pd.DataFrame) else dataset_y

        self.buffer_size = len(self.X) if dataset_indices is None else len(dataset_indices)
        self.indices = np.array(dataset_indices, dtype=int) if dataset_indices is not None else None
        self.iters = 0

    def __getitem__(self, index):
        x = torch.tensor(self.X[index], dtype=torch.float)
        y = torch.tensor(self.y[index], dtype=torch.float) if self.y is not None else None
        return x, y

    def __len__(self):
        return self.buffer_size

class TemplateDataset(BaseDataset):
    def __init__(self, dataset_X, dataset_y, dataset_indices = None,
                 min_seq_len = None, max_seq_len = None, input_wrapper = None, **kwargs):
        super().__init__(dataset_X, dataset_y, dataset_indices)
       
        self.pre_batch_size = kwargs.get('pre_batch_size', PRE_BATCH_SIZE)
        self.buffer_size = self.pre_batch_size * (self.buffer_size//self.pre_batch_size)
        
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.input_wrapper = input_wrapper
        
        self.seed_count = 0
        
        self.use_seq = bool(self.max_seq_len or self.min_seq_len)
        self.min_seq_len = self.min_seq_len or (self.max_seq_len // 2 if self.max_seq_len else 1)
        self.max_seq_len = self.max_seq_len or (2 * self.min_seq_len if self.min_seq_len else 1)

        self.shuffle_indices()
        self.precompute_batch()

    def precompute_batch(self):
        idx = self.iters
        
        batch_size = self.pre_batch_size
        batch_idx = idx // batch_size
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, self.buffer_size)
        batch_indices = self.batch_indices[start_idx:end_idx]

        if self.use_seq:
            window_sizes = np.random.randint(self.min_seq_len, self.max_seq_len + 1, size=len(batch_indices))
            end_indices = batch_indices + window_sizes
            
            assert len(self.X) >= end_indices.any(), "The end index is out of bounds"

            X_batch = [self.X[i:end] for i, end in zip(batch_indices, end_indices)]
            y_batch = [self.y[i:end] for i, end in zip(batch_indices, end_indices)] if self.y is not None else None
        else:
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices] if self.y is not None else None
                            
        if self.input_wrapper is not None:
            X_batch, y_batch = self.input_wrapper(X_batch, y_batch)

        self.cache_X = X_batch
        self.cache_y = y_batch

    def shuffle_indices(self):

        # use numpy function
        set_random_seed(self.seed_count)
        self.seed_count+= 1
        
        permutation_indices = np.random.permutation(self.buffer_size)
        self.batch_indices = permutation_indices if self.indices is None else self.indices[permutation_indices.tolist()]
        self.iters = 0

    def get_item(self, idx):
        pre_batches = min(self.pre_batch_size, len(self.cache_X))
        selected_idx = idx % pre_batches
        x = torch.tensor(self.cache_X[selected_idx], dtype=torch.float)
        y = torch.tensor(self.cache_y[selected_idx], dtype=torch.float) if self.cache_y is not None else None
        self.iters += 1
        return x, y

    def __getitem__(self, idx):
        if self.iters >= self.buffer_size:
            self.shuffle_indices()

        if idx % self.pre_batch_size == 0:
            self.precompute_batch()

        return self.get_item(idx)
