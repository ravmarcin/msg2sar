try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import os
from settings.paths import DATA_DIR
from utils.internal.io.json_io import open_json
from utils.internal.log.logger import get_logger

log = get_logger()


class MLDataConfig:
    """
    Configuration loader for ML data preparation and training.

    Follows the config pattern established by MsgConfig and GnssConfig.
    Manages ML-specific parameters including data augmentation,
    normalization, batch sizes, and model hyperparameters.
    """

    def __init__(self, config_path: str) -> None:
        self.config = open_json(config_path)
        self.__get_data()
        self.__get_paths()
        self.__get_ml_training_config()
        self.__get_ml_model_config()

        log.info(
            f"ML-DATA-CONFIG initialized with: \n"
            f"  Job: {self.job_name}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Input size: {self.input_size}\n"
            f"  Output size: {self.output_size}\n"
            f"  Train/val split: {self.train_val_split}\n"
        )

    def __get_data(self) -> None:
        self.job_name = self.config.get("job_name", "default")
        self.description = self.config.get("description", "")

        self.data = self.config.get("data", {})

        # Reference to other config files
        self.sar_config_path = self.data.get("sar", {}).get("config_path", "")
        self.msg_config_path = self.data.get("msg", {}).get("config_path", "")
        self.gnss_config_path = self.data.get("gnss", {}).get("config_path", "")
        self.gacos_config_path = self.data.get("gacos", {}).get("config_path", "")

        # ML data paths
        self.data_ml = self.data.get("ml", {})

    def __get_paths(self) -> None:
        if self.data_ml:
            self.data_dir = os.path.join(DATA_DIR, self.data_ml['data_child_dir'])
            self.training_folder = os.path.join(self.data_dir, self.data_ml['training_folder'])
            self.validation_folder = os.path.join(self.data_dir, self.data_ml['validation_folder'])
            self.preprocessed_folder = os.path.join(self.data_dir, self.data_ml.get('preprocessed_folder', 'preprocessed'))

            # Create directories if they don't exist
            os.makedirs(self.training_folder, exist_ok=True)
            os.makedirs(self.validation_folder, exist_ok=True)
            os.makedirs(self.preprocessed_folder, exist_ok=True)
        else:
            self.data_dir = None
            self.training_folder = None
            self.validation_folder = None
            self.preprocessed_folder = None

    def __get_ml_training_config(self) -> None:
        ml_training = self.config.get("ml_training", {})

        self.batch_size = ml_training.get("batch_size", 16)
        self.train_val_split = ml_training.get("train_val_split", 0.8)
        self.num_workers = ml_training.get("num_workers", 4)
        self.seed = ml_training.get("seed", 42)

        # Data augmentation
        self.data_augmentation = ml_training.get("data_augmentation", {})
        self.random_flip = self.data_augmentation.get("random_flip", True)
        self.random_rotation = self.data_augmentation.get("random_rotation", 0)
        self.noise_std = self.data_augmentation.get("noise_std", 0.01)

        # Normalization
        self.normalization = ml_training.get("normalization", {})
        self.normalization_method = self.normalization.get("method", "standardize")
        self.per_channel_norm = self.normalization.get("per_channel", True)

        # Input/output sizes
        self.input_size = ml_training.get("input_size", [256, 256])
        self.output_size = ml_training.get("output_size", [128, 128])

    def __get_ml_model_config(self) -> None:
        ml_model = self.config.get("ml_model", {})

        self.architecture = ml_model.get("architecture", "unet")
        self.in_channels = ml_model.get("in_channels", 14)
        self.out_channels = ml_model.get("out_channels", 1)
        self.init_features = ml_model.get("init_features", 64)
        self.model_input_size = ml_model.get("input_size", 256)
        self.model_output_size = ml_model.get("output_size", 128)

        self.loss_function = ml_model.get("loss_function", "mse")

        # Optimizer config
        optimizer_config = ml_model.get("optimizer", {})
        self.optimizer_type = optimizer_config.get("type", "adam")
        self.learning_rate = optimizer_config.get("lr", 0.0001)
        self.weight_decay = optimizer_config.get("weight_decay", 0.0001)

        # Scheduler config
        scheduler_config = ml_model.get("scheduler", {})
        self.scheduler_type = scheduler_config.get("type", "reduce_on_plateau")
        self.scheduler_patience = scheduler_config.get("patience", 5)
        self.scheduler_factor = scheduler_config.get("factor", 0.5)
        self.scheduler_min_lr = scheduler_config.get("min_lr", 1e-7)

        # Training config
        training_config = ml_model.get("training", {})
        self.epochs = training_config.get("epochs", 100)
        self.early_stopping_patience = training_config.get("early_stopping_patience", 10)
        self.checkpoint_dir = os.path.join(self.data_dir, training_config.get("checkpoint_dir", "checkpoints"))
        self.log_dir = os.path.join(self.data_dir, training_config.get("log_dir", "logs"))
        self.save_best_only = training_config.get("save_best_only", True)
        self.checkpoint_frequency = training_config.get("checkpoint_frequency", 5)

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
