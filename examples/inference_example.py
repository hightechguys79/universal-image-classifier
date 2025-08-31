"""
Example inference script for Universal Image Classifier
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import UniversalImageClassifier, load_classifier, quick_predict
from utils import evaluate_model, plot_confusion_matrix, save_predictions
import torch
from PIL import Image
import numpy as np
 
def create_sample_images():
    """Create sample images for demonstration"""
    os.makedirs("sample_images", exist_ok=True)
    
    # Create sample images with different patterns
    for i in range(4):
        # Create different colored images as samples
        if i == 0:  # Red-ish image
            img_array = np.random.randint(150, 255, (64, 64, 3), dtype=np.uint8)
            img_array[:, :, 1:] = np.random.randint(0, 100, (64, 64, 2), dtype=np.uint8)
        elif i == 1:  # Green-ish image
            img_array = np.random.randint(0, 100, (64, 64, 3), dtype=np.uint8)
            img_array[:, :, 1] = np.random.randint(150, 255, (64, 64), dtype=np.uint8)
        elif i == 2:  # Blue-ish image
            img_array = np.random.randint(0, 100, (64, 64, 3), dtype=np.uint8)
            img_array[:, :, 2] = np.random.randint(150, 255, (64, 64), dtype=np.uint8)
        else:  # Mixed image
            img_array = np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)
        
        img = Image.fromarray(img_array)
        img.save(f"sample_images/sample_{i}.jpg")
    
    print("Created sample images in 'sample_images/' directory")

def main():
    """Example inference workflow"""
    
    # Check if model exists
    model_path = "outputs/mlp_deep_residual_best.pth"
    config_path = "outputs/mlp_deep_residual_config.json"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run train_example.py first to train a model")
        return
    
    # Create sample images for demonstration
    create_sample_images()
    
    # Define class names (should match your training data)
    class_names = ["cat", "dog"]  # Update based on your actual classes
    
    # Load classifier
    print("Loading classifier...")
    classifier = load_classifier(
        model_path=model_path,
        config_path=config_path,
        class_names=class_names
    )
    
    print(f"Model loaded successfully!")
    print(f"Device: {classifier.device}")
    print(f"Classes: {classifier.class_names}")
    
    # Single image prediction
    print("\n" + "="*50)
    print("SINGLE IMAGE PREDICTION")
    print("="*50)
    
    sample_image = "sample_images/sample_0.jpg"
    if os.path.exists(sample_image):
        result = classifier.predict(sample_image)
        
        print(f"Image: {sample_image}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Class ID: {result['predicted_class_id']}")
        
        print("\nAll Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
    
    # Top-k predictions
    print("\n" + "="*50)
    print("TOP-K PREDICTIONS")
    print("="*50)
    
    if os.path.exists(sample_image):
        top_k = classifier.get_top_k_predictions(sample_image, k=3)
        
        print(f"Top 3 predictions for {sample_image}:")
        for i, pred in enumerate(top_k, 1):
            print(f"  {i}. {pred['class_name']}: {pred['probability']:.4f}")
    
    # Batch prediction
    print("\n" + "="*50)
    print("BATCH PREDICTION")
    print("="*50)
    
    sample_images = [f"sample_images/sample_{i}.jpg" for i in range(4) 
                    if os.path.exists(f"sample_images/sample_{i}.jpg")]
    
    if sample_images:
        batch_results = classifier.predict_batch(sample_images, batch_size=2)
        
        print(f"Batch prediction results for {len(sample_images)} images:")
        for img_path, result in zip(sample_images, batch_results):
            print(f"  {os.path.basename(img_path)}: {result['predicted_class']} ({result['confidence']:.4f})")
    
    # Quick prediction (convenience function)
    print("\n" + "="*50)
    print("QUICK PREDICTION")
    print("="*50)
    
    if os.path.exists(sample_image):
        quick_result = quick_predict(model_path, sample_image, class_names)
        print(f"Quick prediction: {quick_result['predicted_class']} ({quick_result['confidence']:.4f})")
    
    # Model information
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50) 
    
    from models import count_parameters, calculate_model_size_mb
    
    param_count = count_parameters(classifier.model)
    model_size = calculate_model_size_mb(classifier.model)
    
    print(f"Model Architecture: {classifier.model.__class__.__name__}")
    print(f"Total Parameters: {param_count:,}")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Input Size: {classifier.config.input_height}x{classifier.config.input_width}")
    print(f"Number of Classes: {classifier.config.num_classes}")
    
    print("\n" + "="*50)
    print("INFERENCE COMPLETE")
    print("="*50)
    print("You can now use this model for your own images!")
    print("Simply replace the sample images with your own data.")

if __name__ == "__main__":
    main()
