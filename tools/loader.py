import os
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

def save_model(model_path, model_name, model, opt_model, scheduler_model):
    torch.save(model.state_dict(),os.path.join(model_path, model_name + '.pth'))
    torch.save(opt_model.state_dict(),os.path.join(model_path, 'opt_' + model_name + '.pth'))
    torch.save(scheduler_model.state_dict(),os.path.join(model_path, 'sch_' + model_name + '.pth'))

def load_model(model_path, model_name, model, opt_model, scheduler_model):
    model.load_state_dict(torch.load(model_path + model_name + '.pth', map_location="cuda:0"))
    opt_model.load_state_dict(torch.load(model_path + 'opt_'+ model_name + '.pth', map_location="cuda:0"))
    opt_model.param_groups[0]['capturable'] = True
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
    # Unzip the batch data
    X, y = zip(*batch)
    # Pad sequences so they are all the same length as the longest sequence
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)  # Assumes 0 is an appropriate padding value
    y_padded = pad_sequence(y, batch_first=True, padding_value=-1)  # Use -1 or another flag value for labels, if necessary
    return X_padded, y_padded

def get_data_loader(dataset, batch_size, shuffle = False, num_workers = 0, collate=collate_fn):
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, collate_fn=collate, drop_last=True)

def get_eval_loader(evalset, batch_size, shuffle = False, num_workers = 0, collate = collate_fn):
    return torch.utils.data.DataLoader(evalset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, collate_fn=collate, drop_last=True)

def get_test_loader(testset, batch_size, num_workers=0, collate=collate_fn):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, drop_last=True)

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
    