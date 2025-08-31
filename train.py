"""
Universal training pipeline for image classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple
import wandb
from tqdm import tqdm

from models import create_model, ClassificationLoss
from config import ModelConfig, TrainingConfig, DataConfig

class CustomImageDataset(Dataset):
    """Generic image dataset class"""
    
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[int], 
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class UniversalTrainer:
    """Universal trainer for image classification models"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: str = "outputs"
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create training and validation transforms"""
        base_transform = [
            transforms.Resize(self.data_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.data_config.normalize_mean,
                std=self.data_config.normalize_std
            )
        ]
        
        train_transform = transforms.Compose([
            transforms.Resize(self.data_config.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.data_config.normalize_mean,
                std=self.data_config.normalize_std
            )
        ] if self.data_config.augment_training else base_transform)
        
        val_transform = transforms.Compose(base_transform)
        
        return train_transform, val_transform
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        model_name: str = "mlp_deep_residual",
        use_wandb: bool = False,
        project_name: str = "universal-image-classifier"
    ):
        """Train the model""" 
         
        if use_wandb:
            wandb.init(project=project_name, config={
                **self.model_config.__dict__,
                **self.training_config.__dict__,
                **self.data_config.__dict__
            })
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        # Create model
        model = create_model(model_name, self.model_config)
        model.to(self.device)
        
        # Loss and optimizer
        criterion = ClassificationLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.training_config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.training_config.num_epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            val_accuracy = 0.0
            if val_loader:
                model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_accuracy = 100 * val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
                
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self.save_model(model, model_name, epoch, avg_val_loss)
                else:
                    patience_counter += 1
                
                if patience_counter >= self.training_config.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Logging
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            if val_loader:
                print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            if use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_accuracy,
                }
                if val_loader:
                    log_dict.update({
                        "val_loss": avg_val_loss,
                        "val_accuracy": val_accuracy
                    })
                wandb.log(log_dict)
        
        if use_wandb:
            wandb.finish()
    
    def save_model(self, model: nn.Module, model_name: str, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_type': model_name,
            'state_dict': model.state_dict(),
            'val_loss': val_loss,
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__
        }
        
        model_path = self.output_dir / f"{model_name}_best.pth"
        torch.save(checkpoint, model_path)
        
        # Also save config as JSON
        config_path = self.output_dir / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.model_config.__dict__, f, indent=2)
        
        print(f"Model saved to {model_path}")
