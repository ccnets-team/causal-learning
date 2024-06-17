import os
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

def setup_directories(base_path = '../../'):
    set_model_path = os.path.join(base_path, "models")
    set_temp_path = os.path.join(base_path, "models/temp")
    set_log_path = os.path.join(base_path, "logs")

    for path in [set_model_path, set_temp_path, set_log_path]:
        os.makedirs(path, exist_ok=True)

    return set_model_path, set_temp_path, set_log_path
    
def save_model(model_path, model_name, model, opt_models, scheduler_models):
    torch.save(model.state_dict(),os.path.join(model_path, model_name + '.pth'))
    torch.save(opt_models.state_dict(),os.path.join(model_path, 'opt_' + model_name + '.pth'))
    torch.save(scheduler_models.state_dict(),os.path.join(model_path, 'sch_' + model_name + '.pth'))

def load_model(model_path, model_name, model, opt_model, scheduler_model):
    if model is not None:
        model.load_state_dict(torch.load(model_path + model_name + '.pth', map_location="cuda:0"))
    
    if opt_model is not None:
        opt_model.load_state_dict(torch.load(model_path +  'opt_'+ model_name + '.pth', map_location="cuda:0"))
        opt_model.param_groups[0]['capturable'] = True

    if scheduler_model is not None:
        scheduler_model.load_state_dict(torch.load(model_path + 'sch_'+ model_name + '.pth', map_location="cuda:0"))

def save_dataset(trainset, testset, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(trainset,path + "trainset.pt")
    torch.save(testset, path + "testset.pt")

def load_dataset(path):
    if not os.path.isdir(path):
        raise Exception(f"No such Path : {path}")
    trainset = torch.load(path + "trainset.pt")
    testset = torch.load(path + "testset.pt")
    return trainset, testset

def collate_fn(batch):
    X, y = zip(*batch)
    # Directly use the tensors from X if they are already tensors, else convert appropriately
    X_padded = pad_sequence([x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in X], batch_first=True, padding_value=0)
    
    if any(label is None for label in y):
        y_padded = None
    else:
        # Directly use the tensors from y if they are already tensors, else convert appropriately
        y_padded = pad_sequence([label.clone().detach() if isinstance(label, torch.Tensor) else torch.tensor(label) for label in y], batch_first=True, padding_value=-1)
    
    return X_padded, y_padded

def get_data_loader(dataset, batch_size, num_workers = 0, collate=collate_fn):
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, collate_fn=collate, drop_last=True)

def get_eval_loader(evalset, batch_size, num_workers = 0, collate = collate_fn):
    return torch.utils.data.DataLoader(evalset, batch_size = batch_size, shuffle = False, num_workers = num_workers, collate_fn=collate, drop_last=True)

def get_test_loader(testset, batch_size, num_workers=0, collate=collate_fn):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, drop_last=False)

def load_trainer(self, ccnet_network = True):
    if ccnet_network:
        _load_trainer(self.helper.model_path, self.ccnet_trainer)
    else:
        _load_trainer(self.helper.model_path, self.encoder_trainer)

def save_trainer(model_path, trainer):
    # Lists of components to be saved for GPT
    network_names = trainer.network_names
    networks = trainer.networks
    optimizers = trainer.optimizers
    schedulers = trainer.schedulers
    
    # Ensure all lists are synchronized in length
    assert len(network_names) == len(networks) == len(optimizers) == len(schedulers), "model component lists must be of the same length"
    
    # Iterate over each model component set and save
    for model_name, network, optimizer, scheduler in zip(network_names, networks, optimizers, schedulers):
        save_model(model_path, model_name, network, optimizer, scheduler)

def _load_trainer(model_path, trainer):
    # Lists of components to be loadd for GPT
    network_names = trainer.network_names
    networks = trainer.networks
    optimizers = trainer.optimizers
    schedulers = trainer.schedulers
    
    # Ensure all lists are synchronized in length
    assert len(network_names) == len(networks) == len(optimizers) == len(schedulers), "model component lists must be of the same length"
    
    # Iterate over each model component set and load
    for model_name, network, optimizer, scheduler in zip(network_names, networks, optimizers, schedulers):
        load_model(model_path + '/', model_name, network, optimizer, scheduler)
    