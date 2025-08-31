"""
Example training script for Universal Image Classifier
"""
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import UniversalTrainer, CustomImageDataset
from config import ModelConfig, TrainingConfig, DataConfig
from utils import load_dataset_from_folder, visualize_dataset, plot_class_distribution
import torch
from torch.utils.data import random_split

def main():
    """Example training workflow"""
    
    # Configuration
    model_config = ModelConfig(
        input_height=64,
        input_width=64,
        num_classes=4,  # Adjust based on your dataset
        hidden_dim=256,
        num_layers=6,
        dropout_rate=0.1,
        use_batch_norm=True
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50,
        weight_decay=1e-4,
        early_stopping_patience=10,
        validation_split=0.2
    )
    
    data_config = DataConfig(
        image_size=(64, 64),
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        augment_training=True
    )
    
    # Load dataset (replace with your data path)
    data_dir = "path/to/your/dataset"  # Structure: data_dir/class1/, data_dir/class2/, etc.
    
    try:
        image_paths, labels, class_names = load_dataset_from_folder(data_dir)
        print(f"Loaded dataset with {len(class_names)} classes: {class_names}")
        
        # Visualize dataset
        visualize_dataset(image_paths, labels, class_names, num_samples=16)
        plot_class_distribution(labels, class_names)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy dataset for demonstration...")
        
        # Create dummy data for demonstration
        import torch
        from PIL import Image
        import numpy as np
        
        # Create dummy images and save them
        os.makedirs("dummy_data/cat", exist_ok=True)
        os.makedirs("dummy_data/dog", exist_ok=True)
        
        for i in range(20):
            # Create random RGB images
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            if i < 10:
                img.save(f"dummy_data/cat/cat_{i}.jpg")
            else:
                img.save(f"dummy_data/dog/dog_{i-10}.jpg")
        
        image_paths, labels, class_names = load_dataset_from_folder("dummy_data")
    
    # Update model config with actual number of classes
    model_config.num_classes = len(class_names)
    
    # Create trainer
    trainer = UniversalTrainer(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        output_dir="outputs"
    )
    
    # Create transforms
    train_transform, val_transform = trainer.create_transforms()
    
    # Create dataset
    full_dataset = CustomImageDataset(image_paths, labels, train_transform)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Train model
    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name="mlp_deep_residual",
        use_wandb=False,  # Set to True if you want to use Weights & Biases
        project_name="universal-image-classifier-demo"
    )
    
    print("Training completed!")
    print("Model saved in 'outputs/' directory")
    print("You can now use the trained model for inference with inference_example.py")

if __name__ == "__main__":
    main()
