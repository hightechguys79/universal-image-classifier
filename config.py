"""
Configuration classes for the Universal Image Classifier
"""
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass 
class ModelConfig:
    """Configuration for model architecture"""
    input_height: int = 64 
    input_width: int = 64
    num_classes: int = 10
    hidden_dim: int = 128
    num_layers: int = 4
    dropout_rate: float = 0.1
    use_batch_norm: bool = True

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
@dataclass
class DataConfig:
    """Configuration for data processing"""
    image_size: Tuple[int, int] = (64, 64)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    augment_training: bool = True
