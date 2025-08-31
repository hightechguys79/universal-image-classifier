"""
Utility functions for the Universal Image Classifier
"""
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd

def load_dataset_from_folder(
    data_dir: str, 
    class_names: Optional[List[str]] = None,
    valid_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
) -> Tuple[List[str], List[int], List[str]]:
    """
    Load dataset from folder structure:
    data_dir/
    ├── class1/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── class2/
        ├── img3.jpg
        └── img4.jpg
    
    Returns:
        image_paths: List of image file paths
        labels: List of corresponding class indices
        class_names: List of class names
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    # Get class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    class_dirs.sort()
    
    if class_names is None:
        class_names = [d.name for d in class_dirs]
    
    image_paths = []
    labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        if class_name not in class_names:
            continue
            
        class_label = class_names.index(class_name)
        
        # Get all image files in this class directory
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                image_paths.append(str(img_file))
                labels.append(class_label)
    
    print(f"Loaded {len(image_paths)} images from {len(class_names)} classes")
    for i, class_name in enumerate(class_names):
        count = labels.count(i)
        print(f"  {class_name}: {count} images")
    
    return image_paths, labels, class_names

def load_dataset_from_csv(
    csv_file: str,
    image_col: str = 'image_path',
    label_col: str = 'label',
    class_names: Optional[List[str]] = None
) -> Tuple[List[str], List[int], List[str]]:
    """
    Load dataset from CSV file with columns for image paths and labels
    
    Args:
        csv_file: Path to CSV file
        image_col: Column name for image paths
        label_col: Column name for labels
        class_names: Optional list of class names
    
    Returns:
        image_paths: List of image file paths
        labels: List of corresponding class indices
        class_names: List of class names
    """
    df = pd.read_csv(csv_file)
    
    image_paths = df[image_col].tolist()
    
    # Handle both string and numeric labels
    if df[label_col].dtype == 'object':  # String labels
        unique_labels = sorted(df[label_col].unique())
        if class_names is None:
            class_names = unique_labels
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        labels = [label_to_idx[label] for label in df[label_col]]
    else:  # Numeric labels
        labels = df[label_col].tolist()
        if class_names is None:
            class_names = [f"class_{i}" for i in range(max(labels) + 1)]
    
    print(f"Loaded {len(image_paths)} images from CSV")
    return image_paths, labels, class_names

def visualize_dataset(
    image_paths: List[str], 
    labels: List[int], 
    class_names: List[str],
    num_samples: int = 16,
    figsize: Tuple[int, int] = (12, 8)
):
    """Visualize random samples from the dataset"""
    import random
    
    # Sample random images
    indices = random.sample(range(len(image_paths)), min(num_samples, len(image_paths)))
    
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        img = Image.open(image_paths[idx]).convert('RGB')
        axes[i].imshow(img)
        axes[i].set_title(f"{class_names[labels[idx]]}")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_class_distribution(labels: List[int], class_names: List[str]):
    """Plot the distribution of classes in the dataset"""
    class_counts = [labels.count(i) for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_counts)
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str = "cpu"
) -> Dict:
    """Evaluate model performance and generate metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    # Classification report
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'true_labels': all_labels,
        'classification_report': report
    }

def plot_confusion_matrix(
    true_labels: List[int],
    predictions: List[int],
    class_names: List[str],
    figsize: Tuple[int, int] = (8, 6)
):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def save_predictions(
    predictions: List[int],
    image_paths: List[str],
    class_names: List[str],
    output_file: str
):
    """Save predictions to CSV file"""
    df = pd.DataFrame({
        'image_path': image_paths,
        'predicted_class_id': predictions,
        'predicted_class_name': [class_names[pred] for pred in predictions]
    })
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def create_model_card(
    model_name: str,
    model_config: Dict,
    training_config: Dict,
    performance_metrics: Dict,
    class_names: List[str],
    dataset_info: str,
    output_file: str = "MODEL_CARD.md"
):
    """Create a model card for Hugging Face"""
    
    card_content = f"""---
license: mit
tags:
- image-classification
- pytorch
- computer-vision
datasets:
- custom
metrics:
- accuracy
library_name: pytorch
---

# Universal Image Classifier - {model_name}

## Model Description

This is a {model_name} model trained using the Universal Image Classifier framework. It's designed for flexible image classification tasks and can be easily adapted to different datasets.

## Model Architecture

- **Model Type**: {model_name}
- **Input Size**: {model_config.get('input_height', 64)}x{model_config.get('input_width', 64)}
- **Number of Classes**: {model_config.get('num_classes', len(class_names))}
- **Hidden Dimensions**: {model_config.get('hidden_dim', 128)}
- **Number of Layers**: {model_config.get('num_layers', 4)}
- **Dropout Rate**: {model_config.get('dropout_rate', 0.1)}
- **Batch Normalization**: {model_config.get('use_batch_norm', True)}

## Training Configuration

- **Batch Size**: {training_config.get('batch_size', 32)}
- **Learning Rate**: {training_config.get('learning_rate', 0.001)}
- **Weight Decay**: {training_config.get('weight_decay', 1e-4)}
- **Early Stopping Patience**: {training_config.get('early_stopping_patience', 10)}

## Performance

- **Accuracy**: {performance_metrics.get('accuracy', 'N/A'):.4f}
- **Classes**: {', '.join(class_names)}

## Dataset

{dataset_info}

## Usage

```python
from inference import load_classifier

# Load the model
classifier = load_classifier(
    model_path="path/to/model.pth",
    config_path="path/to/config.json",
    class_names={class_names}
)

# Make predictions
result = classifier.predict("path/to/image.jpg")
print(f"Predicted: {{result['predicted_class']}} ({{result['confidence']:.3f}})")
```

## Training

This model was trained using the Universal Image Classifier framework. To train your own model:

```python
from train import UniversalTrainer
from config import ModelConfig, TrainingConfig, DataConfig

# Configure and train
trainer = UniversalTrainer(model_config, training_config, data_config)
trainer.train(train_dataset, val_dataset, model_name="{model_name}")
```

## Framework

Built with the Universal Image Classifier framework - a flexible PyTorch-based solution for image classification tasks.

## License

MIT License
"""
    
    with open(output_file, 'w') as f:
        f.write(card_content)
    
    print(f"Model card saved to {output_file}")

def export_to_huggingface(
    model_path: str,
    config_path: str,
    class_names: List[str],
    model_name: str,
    output_dir: str = "huggingface_model"
):
    """Export model for Hugging Face Hub"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Copy model files
    import shutil
    shutil.copy(model_path, output_path / "pytorch_model.pth")
    shutil.copy(config_path, output_path / "config.json")
    
    # Create class names file
    with open(output_path / "class_names.json", 'w') as f:
        json.dump(class_names, f, indent=2)
    
    # Create inference script
    inference_script = '''
from inference import UniversalImageClassifier
import json

def load_model():
    """Load the model for inference"""
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    
    classifier = UniversalImageClassifier(
        model_path="pytorch_model.pth",
        config_path="config.json",
        class_names=class_names
    )
    return classifier

def predict(image_path):
    """Predict image class"""
    classifier = load_model()
    return classifier.predict(image_path)
'''
    
    with open(output_path / "inference_hf.py", 'w') as f:
        f.write(inference_script)
    
    print(f"Model exported to {output_path} for Hugging Face Hub")
