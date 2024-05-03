class DataConfig:
    def __init__(self, dataset_name:str, task_type:str, obs_shape:list, label_size):
        valid_task_types = ['classification', 'regression', 'binary']
        self.dataset_name = dataset_name
        if task_type not in valid_task_types:
            raise ValueError(f"Invalid task_type '{task_type}'. Valid options are {valid_task_types}")

        self.task_type = task_type
        
        self.obs_shape = obs_shape
        self.label_size = label_size
        
    def initialize_(self, d_model:int):
            
        self.stoch_size = max(d_model, 1)
        self.det_size = max(d_model, 1)
        
        self.state_size = self.stoch_size + self.det_size
        
        self.explain_size = max(d_model, 1)
        