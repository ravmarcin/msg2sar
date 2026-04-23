try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any
import numpy as np
from utils.internal.log.logger import get_logger
from utils.internal.ml.data_config import MLDataConfig

log = get_logger()


class AtmosphericCorrectionTrainer:
    """
    Training infrastructure for atmospheric correction model.

    Supports:
    - Apple Silicon (MPS) for Mac M chips
    - CUDA for NVIDIA GPUs
    - CPU fallback
    - TensorBoard logging
    - Checkpointing
    - Early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        config: MLDataConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            config: MLDataConfig instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto-detect)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup device (Mac M chips support)
        self.device = self._setup_device(device)
        log.info(f"Using device: {self.device}")

        # Move model to device
        self.model = model.to(self.device)

        # Setup loss function
        self.criterion = self._setup_loss()

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()

        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        log.info(
            f"Trainer initialized:\n"
            f"  Device: {self.device}\n"
            f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\n"
            f"  Optimizer: {self.config.optimizer_type}\n"
            f"  Loss: {self.config.loss_function}\n"
            f"  Epochs: {self.config.epochs}\n"
            f"  Batch size: {self.config.batch_size}"
        )

    def _setup_device(self, device: Optional[str] = None) -> torch.device:
        """
        Setup compute device with Apple Silicon (MPS) support.

        Priority:
        1. User-specified device
        2. MPS (Apple Silicon M chips)
        3. CUDA (NVIDIA GPUs)
        4. CPU (fallback)

        Args:
            device: Optional device string ('mps', 'cuda', 'cpu')

        Returns:
            torch.device
        """
        if device is not None:
            return torch.device(device)

        # Auto-detect best available device
        if torch.backends.mps.is_available():
            log.info("Apple Silicon (MPS) detected and available")
            return torch.device("mps")
        elif torch.cuda.is_available():
            log.info("CUDA detected and available")
            return torch.device("cuda")
        else:
            log.warning("No GPU detected, using CPU")
            return torch.device("cpu")

    def _setup_loss(self) -> nn.Module:
        """Setup loss function."""
        if self.config.loss_function == "mse":
            return nn.MSELoss()
        elif self.config.loss_function == "mae" or self.config.loss_function == "l1":
            return nn.L1Loss()
        elif self.config.loss_function == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            log.warning(f"Unknown loss function '{self.config.loss_function}', using MSE")
            return nn.MSELoss()

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        if self.config.optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            log.warning(f"Unknown optimizer '{self.config.optimizer_type}', using Adam")
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.scheduler_type.lower() == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr
            )
        elif self.config.scheduler_type.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler_type.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor
            )
        else:
            log.info("No scheduler specified")
            return None

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Log progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                log.debug(
                    f"Epoch {self.current_epoch} [{batch_idx+1}/{num_batches}] "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, targets)

                # Accumulate loss
                total_loss += loss.item()

        avg_loss = total_loss / num_batches

        return {
            'val_loss': avg_loss
        }

    def train(self) -> Dict[str, Any]:
        """
        Complete training loop with validation and checkpointing.

        Returns:
            Training history dictionary
        """
        log.info(f"Starting training for {self.config.epochs} epochs...")
        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            # Train one epoch
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['val_loss']

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(current_lr)

            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, self.current_epoch)
            self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, self.current_epoch)

            epoch_time = time.time() - epoch_start

            log.info(
                f"Epoch {self.current_epoch}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model
                self.save_checkpoint('best_model.pth', is_best=True)
                log.info(f"✓ New best model saved (val_loss: {val_loss:.6f})")

            else:
                self.epochs_without_improvement += 1

                # Check early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    log.info(
                        f"Early stopping triggered after {self.current_epoch} epochs "
                        f"({self.config.early_stopping_patience} epochs without improvement)"
                    )
                    break

            # Periodic checkpoint
            if self.current_epoch % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pth')

        total_time = time.time() - start_time
        log.info(
            f"Training completed!\n"
            f"  Total time: {total_time/60:.2f} minutes\n"
            f"  Best val loss: {self.best_val_loss:.6f}\n"
            f"  Final epoch: {self.current_epoch}"
        )

        # Save final model
        self.save_checkpoint('final_model.pth')

        # Close TensorBoard writer
        self.writer.close()

        return self.training_history

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': {
                'in_channels': self.model.in_channels,
                'out_channels': self.model.out_channels,
                'init_features': self.model.init_features,
                'input_size': self.model.input_size,
                'output_size': self.model.output_size
            }
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)

        if not is_best:
            log.debug(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)

        if not os.path.exists(checkpoint_path):
            log.error(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        log.info(
            f"Checkpoint loaded: {checkpoint_path}\n"
            f"  Epoch: {self.current_epoch}\n"
            f"  Best val loss: {self.best_val_loss:.6f}"
        )

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.

        Args:
            inputs: Input tensor [Batch, 14, 256, 256]

        Returns:
            Predictions [Batch, 1, 128, 128]
        """
        self.model.eval()

        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)

        return outputs
