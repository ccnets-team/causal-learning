'''
Author:
        
        PARK, JunHo, junho@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import torch
import pandas as pd
NUM_PRE_BATCHES = 64
MAX_SEQ_LEN = 128

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_X, dataset_y):
        if isinstance(dataset_X, pd.DataFrame): 
            self.X = dataset_X.iloc[:]
        else:
            self.X = dataset_X
        
        if dataset_y is not None and isinstance(dataset_y, pd.DataFrame):
            self.y = dataset_y.iloc[:]
        else:
            self.y = dataset_y
                
        self.dataset_length = len(self.X)
        self.total_iters = 0
            
        self.X_cache = None
        self.y_cache = None
        
        if self.y is not None:
            assert(len(self.X) == len(self.y))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        return self.X[index], self.y[index] if self.y is not None else None
    
class TemplateDataset(BaseDataset):
    def __init__(self, dataset_X, dataset_y, **kwargs):
        BaseDataset.__init__(self, dataset_X, dataset_y)
       
        self.max_seq_len = kwargs.get('max_seq_len', None)
        self.min_seq_len = kwargs.get('min_seq_len', None)
        self.pre_batches = kwargs.get('pre_batches', NUM_PRE_BATCHES)
        self.input_wrapper = kwargs.get('input_wrapper', None)
        
        self.use_seq = False
        if self.max_seq_len is not None or self.min_seq_len is not None:
            self.use_seq = True
            if self.min_seq_len is None:
                self.min_seq_len = self.max_seq_len//2
            if self.max_seq_len is None:
                self.max_seq_len = 2 * self.min_seq_len
        
        self.shuffle_indices()
        self.precompute_batches(self.total_iters)

    def precompute_batches(self, idx):
        if idx % self.pre_batches != 0:
            return 
        pre_batches = self.pre_batches
        batch_idx = idx // pre_batches
        start_idx = batch_idx * pre_batches
        end_idx = min(start_idx + pre_batches, len(self.X))
        batch_indices = self.batch_indices[start_idx:end_idx]

        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        if self.input_wrapper is not None:
            X_batch, y_batch = self.input_wrapper(X_batch, y_batch)

        self.X_cache = X_batch
        self.y_cache = y_batch

    def get_item(self, idx):
        pre_batches = min(self.pre_batches, len(self.X_cache))
        selected_idx = idx % pre_batches
        return self.X_cache[selected_idx], self.y_cache[selected_idx] if self.y_cache is not None else None

    def shuffle_indices(self):
        if self.total_iters % self.dataset_length != 0:
            return
        self.batch_indices = torch.randperm(self.dataset_length)
        self.total_iters = 0

    def __getitem__(self, idx):
        self.precompute_batches(self.total_iters)
        self.shuffle_indices()

        self.total_iters += 1

        return self.get_item(idx)
