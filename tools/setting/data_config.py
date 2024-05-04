class DataConfig:
    """
    Configuration class for datasets used in various machine learning tasks, supporting 
    different classification schemes and regression.

    Attributes:
        dataset_name (str): Name of the dataset, e.g., 'CelebA', 'MNIST'.
        task_type (str): Type of ML task; valid options are 'binary_classification', 
                         'multi_class_classification', 'multi_label_classification', 'regression'.
        obs_shape (list): Shape of the observations, typically [channels, height, width] for images.
        label_size (int): Number of labels or output size, specifically:
            - For 'binary_classification': 1 (a single sigmoid output for two classes).
            - For 'multi_class_classification': Equal to the number of classes (one output per class using softmax).
            - For 'multi_label_classification': Equal to the number of possible labels (multiple sigmoid outputs).
            - For 'regression': Typically 1 for single target regression, more for multiple regression targets.
        show_image_indices (list, optional): Indices of images to show for visualization; defaults to None.
    """
    def __init__(self, dataset_name: str, task_type: str, obs_shape: list, label_size: int, show_image_indices: list = None):
        valid_task_types = ['binary_classification', 'multi_class_classification', 
                            'multi_label_classification', 'regression']
        if task_type not in valid_task_types:
            raise ValueError(f"Invalid task_type '{task_type}'. Valid options are {valid_task_types}")

        self.dataset_name = dataset_name
        self.task_type = task_type
        self.obs_shape = obs_shape
        self.label_size = label_size
        self.show_image_indices = show_image_indices or []

    def __repr__(self):
        """
        Provides a string representation of the DataConfig object for debugging and logging.
        """
        return (f"DataConfig(dataset_name={self.dataset_name}, task_type={self.task_type}, "
                f"obs_shape={self.obs_shape}, label_size={self.label_size}, "
                f"show_image_indices={self.show_image_indices})")
