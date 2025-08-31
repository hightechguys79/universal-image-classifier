"""
Universal inference pipeline for image classification
""" 
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Union, Optional
import json
from pathlib import Path

from models import create_model, MODEL_REGISTRY
from config import ModelConfig 

class UniversalImageClassifier:
    """Universal image classifier with flexible configuration"""
    
    def __init__(
        self, 
        model_path: str, 
        config_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the universal classifier
        
        Args:
            model_path: Path to the model weights (.pth file)
            config_path: Path to the model config (.json file)
            class_names: List of class names (if None, uses numeric labels)
            device: Device to run inference on (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = ModelConfig(**config_dict)
        else:
            self.config = ModelConfig()
        
        # Set up class names
        self.class_names = class_names or [f"class_{i}" for i in range(self.config.num_classes)]
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Set up preprocessing
        self.transform = self._create_transform()
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model type from checkpoint or filename
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
            state_dict = checkpoint['state_dict']
        else:
            # Try to infer from filename
            model_path_obj = Path(model_path)
            model_type = model_path_obj.stem.replace('_model', '')
            if model_type not in MODEL_REGISTRY:
                model_type = 'mlp_deep_residual'  # Default
            state_dict = checkpoint
        
        model = create_model(model_type, self.config)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def _create_transform(self) -> transforms.Compose:
        """Create preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((self.config.input_height, self.config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Predict the class of a single image
        
        Args:
            image: Image as file path, PIL Image, or numpy array
            
        Returns:
            Dictionary with prediction results
        """
        # Convert input to PIL Image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError("Image must be file path, PIL Image, or numpy array")
        
        # Preprocess
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        return {
            'predicted_class': self.class_names[predicted_class_idx],
            'predicted_class_id': predicted_class_idx,
            'confidence': confidence,
            'all_probabilities': {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'raw_logits': logits[0].cpu().numpy().tolist()
        }
    
    def predict_batch(self, images: List, batch_size: int = 32) -> List[Dict]:
        """
        Predict classes for a batch of images
        
        Args:
            images: List of images (various formats supported)
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = [self.predict(img) for img in batch]
            results.extend(batch_results)
        return results
    
    def get_top_k_predictions(self, image: Union[str, Image.Image, np.ndarray], k: int = 3) -> List[Dict]:
        """Get top-k predictions for an image"""
        result = self.predict(image)
        probabilities = result['all_probabilities']
        
        # Sort by probability
        sorted_predictions = sorted(
            probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [
            {
                'class_name': class_name,
                'probability': prob,
                'class_id': self.class_names.index(class_name)
            }
            for class_name, prob in sorted_predictions[:k]
        ]

# Convenience functions
def load_classifier(
    model_path: str, 
    config_path: Optional[str] = None,
    class_names: Optional[List[str]] = None
) -> UniversalImageClassifier:
    """Convenience function to load a classifier"""
    return UniversalImageClassifier(model_path, config_path, class_names)

def quick_predict(model_path: str, image_path: str, class_names: Optional[List[str]] = None) -> Dict:
    """Quick prediction for a single image"""
    classifier = load_classifier(model_path, class_names=class_names)
    return classifier.predict(image_path)
