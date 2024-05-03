class DataConfig:
    def __init__(self, dataset_name:str, task_type:str, obs_shape:list, label_size:int, show_image_indices: list = None):
        valid_task_types = ['classification', 'regression', 'binary']
        self.dataset_name = dataset_name
        if task_type not in valid_task_types:
            raise ValueError(f"Invalid task_type '{task_type}'. Valid options are {valid_task_types}")

        self.task_type = task_type
        
        self.obs_shape = obs_shape
        self.label_size = label_size
        self.show_image_indices = show_image_indices