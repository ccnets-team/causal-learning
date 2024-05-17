class DataConfig:
    """
    Configuration class for datasets used in various machine learning tasks, including
    classification schemes and regression models. This class supports configuring data
    specific to different tasks like image recognition, sentiment analysis, and more.

    Attributes:
        dataset_name (str): Name of the dataset, such as 'CelebA' or 'MNIST', indicating
                            which dataset is being configured.
        task_type (str): Specifies the type of machine learning task. Valid options are:
                         'binary_classification', 'multi_class_classification',
                         'multi_label_classification', 'regression', 'ordinal_regression',
                         'encoding', 'augmentation', 'generation', 'reconstruction'.
                         Each type corresponds to a different modeling approach or objective.
        obs_shape (list): Dimensions of the observations (data inputs), commonly used
                          to specify the shape of images, typically as [channels, height, width].
        label_size (int): Defines the size of the output space needed for the task:
            - 'binary_classification': 1 (binary outcome using a sigmoid activation).
            - 'multi_class_classification': Corresponds to the number of distinct classes
                                           (using softmax for multi-class output).
            - 'multi_label_classification': Corresponds to the number of labels for multi-label
                                            binary outcomes (using multiple sigmoid activations).
            - 'regression': Typically 1 for a single continuous target, can be more for
                            multi-dimensional regression outcomes.
            - 'ordinal_regression': Typically 1, used for tasks where the target is ordinal,
                                    classified into ordered categories.
        show_image_indices (list, optional): A list of indices specifying which images to display
                                             for visual inspection or debugging purposes. Defaults to None.
    
    Methods:
        __init__(self, dataset_name, task_type, obs_shape, label_size=None, explain_size=None,
                 state_size=None, show_image_indices=None):
            Initializes a new instance of the DataConfig class. Validates the task type and sets up
            the configuration with the provided values.
        
        __repr__(self):
            Provides a string representation of the DataConfig instance, which is helpful for debugging
            and logging the configuration details.
    """
    def __init__(self, dataset_name: str, task_type: str, obs_shape: list, label_size: int = None,
                 explain_size: int = None, state_size: int = None, show_image_indices: list = None):
        valid_task_types = ['binary_classification', 'multi_class_classification',
                            'multi_label_classification', 'regression', 'ordinal_regression',
                            'encoding', 'augmentation', 'generation', 'reconstruction']
        if task_type not in valid_task_types:
            raise ValueError(f"Invalid task type '{task_type}'. Valid options are {valid_task_types}")
        
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.obs_shape = obs_shape
        self.label_size = label_size
        self.explain_size = explain_size
        self.state_size = state_size
        self.show_image_indices = show_image_indices or []

    def __repr__(self):
        return (f"DataConfig(dataset_name={self.dataset_name}, task_type={self.task_type}, "
                f"obs_shape={self.obs_shape}, label_size={self.label_size}, "
                f"explain_size={self.explain_size}, state_size={self.state_size},"
                f"show_image_indices={self.show_image_indices})")
