class DataConfig:
    """
    Configuration class for managing dataset parameters in various machine learning tasks,
    facilitating the adjustment of network architectures based on the task type and data characteristics.
    This configuration impacts the final layer and function adaptations within the models, especially
    in neural network settings that focus on tasks like image recognition, text processing, or structured data analysis.

    Attributes:
        dataset_name (str): Specifies the dataset, such as 'CelebA' or 'MNIST'.
        task_type (str): Determines the neural model configuration. Supported types include:
                         'binary_classification', 'multi_class_classification', 'multi_label_classification',
                         'regression', 'ordinal_regression', 'compositional_regression'.
        obs_shape (list): Defines the input dimensions appropriate for the data type:
                        - For image data, this is typically specified as [channels, height, width].
                        - For tabular data, specify the number of features as [num_features].
                        Note: While sequence data inherently includes a sequence dimension, do not include this in the `obs_shape`. 
                        Sequence dimensions can vary per batch; hence, the API internally handles the sequence dimension 
                        to accommodate batch-specific variations.
        label_size (int): Specifies the output dimension necessary for the model, varying by task:
                          - Binary outputs use 1 for binary classification.
                          - Multiclass outputs match the number of classes.
                          - Regression tasks typically use 1 but may be higher for multi-dimensional targets.
        explain_size (int, optional): Dimensionality of the latent space from the explainer network, essential for 
                                      generating compressed, efficient explanations during inference and data generation.
                                      Defaults to half of `d_model` if not set.
        show_image_indices (list, optional): Indices of images to be displayed for debugging or visualization purposes.

    Methods:
        __init__(self, dataset_name, task_type, obs_shape, label_size=None, explain_size=None, show_image_indices=None):
            Initializes the DataConfig with the specified dataset characteristics.
            Raises an error if an unsupported task type is specified.
    """
    def __init__(self, dataset_name: str, task_type: str, obs_shape: list, 
                 label_size: int = None, label_scale: list[float] = None, 
                 explain_size: int = None, show_image_indices: list = None):
        valid_task_types = ['binary_classification', 'multi_class_classification','multi_label_classification', 
                            'regression', 'ordinal_regression', 'compositional_regression']
        if task_type not in valid_task_types:
            raise ValueError(f"Invalid task type '{task_type}'. Valid options are {valid_task_types}")
        
        # replace space with '-' and convert to lowercase
        self.dataset_name = dataset_name.replace(' ', '-').lower()
        self.task_type = task_type
        self.obs_shape = obs_shape
        self.label_size = label_size
        self.label_scale = label_scale
        self.explain_size = explain_size
        self.show_image_indices = show_image_indices

    def __repr__(self):
        label_size_repr = f", label_size={self.label_size}" if self.label_size is not None else ""
        explain_size_repr = f", explain_size={self.explain_size}" if self.explain_size is not None else ""
        show_image_indices_repr = f", show_image_indices={self.show_image_indices}" if self.show_image_indices is not None else ""
        return (f"DataConfig(dataset_name={self.dataset_name}, task_type={self.task_type}, "
                f"obs_shape={self.obs_shape}{label_size_repr}{explain_size_repr}{show_image_indices_repr})")
        