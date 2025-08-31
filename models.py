"""
Universal Image Classification Models
Flexible architectures for any image classification task
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from config import ModelConfig

class ClassificationLoss(nn.Module):
    """Multi-class classification loss with label smoothing option"""
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            return F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        return F.cross_entropy(logits, target)

class LinearClassifier(nn.Module):
    """Simple linear baseline classifier"""
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        
        self.config = config
        in_features = 3 * config.input_height * config.input_width
        self.linear = nn.Linear(in_features, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.linear(x)

class MLPClassifier(nn.Module):
    """MLP classifier with configurable architecture"""
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
            
        self.config = config
        in_features = 3 * config.input_height * config.input_width
        
        layers = []
        layers.append(nn.Linear(in_features, config.hidden_dim))
        if config.use_batch_norm:
            layers.append(nn.BatchNorm1d(config.hidden_dim))
        layers.append(nn.ReLU())
        if config.dropout_rate > 0:
            layers.append(nn.Dropout(config.dropout_rate))
        layers.append(nn.Linear(config.hidden_dim, config.num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)

class MLPClassifierDeep(nn.Module):
    """Deep MLP classifier with multiple hidden layers"""
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
            
        self.config = config
        in_features = 3 * config.input_height * config.input_width
        
        layers = []
        # Input layer
        layers.append(nn.Linear(in_features, config.hidden_dim))
        if config.use_batch_norm:
            layers.append(nn.BatchNorm1d(config.hidden_dim))
        layers.append(nn.ReLU())
        if config.dropout_rate > 0:
            layers.append(nn.Dropout(config.dropout_rate))
        
        # Hidden layers
        for _ in range(config.num_layers - 2):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(config.hidden_dim, config.num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)

class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    def __init__(self, hidden_dim: int, use_batch_norm: bool = True, dropout_rate: float = 0.0):
        super().__init__()
        layers = []
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = out + residual  # Residual connection
        return self.relu(out)

class MLPClassifierDeepResidual(nn.Module):
    """Deep MLP with residual connections"""
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
            
        self.config = config
        in_features = 3 * config.input_height * config.input_width
        
        # Input projection
        self.input_layer = nn.Linear(in_features, config.hidden_dim)
        self.input_bn = nn.BatchNorm1d(config.hidden_dim) if config.use_batch_norm else nn.Identity()
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else nn.Identity()
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.hidden_dim, config.use_batch_norm, config.dropout_rate)
            for _ in range(config.num_layers - 2)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        
        # Input projection
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        x = self.input_dropout(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output
        return self.output_layer(x)

# Model factory for easy instantiation
MODEL_REGISTRY = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}

def create_model(model_name: str, config: Optional[ModelConfig] = None) -> nn.Module:
    """Factory function to create models"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](config)

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    return count_parameters(model) * 4 / 1024 / 1024  # 4 bytes per float32
