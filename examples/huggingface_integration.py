"""
Example script for Hugging Face Hub integration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_model_card, export_to_huggingface
from huggingface_hub import HfApi, create_repo, upload_folder
import json

def upload_to_huggingface(
    model_path: str, 
    config_path: str,
    class_names: list,
    model_name: str,
    repo_name: str,
    hf_token: str = None
):
    """
    Upload model to Hugging Face Hub
    
    Args:
        model_path: Path to trained model
        config_path: Path to model config
        class_names: List of class names
        model_name: Model architecture name
        repo_name: Repository name on HF Hub
        hf_token: Hugging Face token (optional if logged in)
    """
    
    # Export model for HF
    export_dir = f"hf_export_{model_name}"
    export_to_huggingface(
        model_path=model_path,
        config_path=config_path,
        class_names=class_names,
        model_name=model_name,
        output_dir=export_dir
    )
     
    # Create model card
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    training_config = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "early_stopping_patience": 10
    }
    
    performance_metrics = {
        "accuracy": 0.95 
    }
    
    dataset_info = """
This model was trained on a custom image classification dataset.
Replace this with information about your specific dataset.
"""
    
    create_model_card(
        model_name=model_name,
        model_config=model_config,
        training_config=training_config,
        performance_metrics=performance_metrics,
        class_names=class_names,
        dataset_info=dataset_info,
        output_file=f"{export_dir}/README.md"
    )
    
    # Create repository and upload
    try:
        api = HfApi(token=hf_token)
        
        # Create repository
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            repo_type="model",
            exist_ok=True
        )
        
        # Upload files
        upload_folder(
            folder_path=export_dir,
            repo_id=repo_name,
            token=hf_token,
            repo_type="model"
        )
      
        print(f"✅ Model successfully uploaded to: https://huggingface.co/ranjeetjha/univeral-img-classifier-small")
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        print("Make sure you have:")
        print("1. Installed huggingface_hub: pip install huggingface_hub")
        print("2. Logged in: huggingface-cli login")
        print("3. Or provided a valid token")

def main():
    """Example HF integration workflow"""
    
    # Check if model exists
    model_path = "outputs/mlp_deep_residual_best.pth"
    config_path = "outputs/mlp_deep_residual_config.json"
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train a model first using train_example.py")
        return
    
    # Configuration
    class_names = ["cat", "dog"]  # Update with your classes
    model_name = "mlp_deep_residual"
    repo_name = "your-username/universal-image-classifier-demo"  # Update with your username
    
    print("🚀 Preparing model for Hugging Face Hub...")
    
    # Export model (this works without HF token)
    export_to_huggingface(
        model_path=model_path,
        config_path=config_path,
        class_names=class_names,
        model_name=model_name,
        output_dir="hf_export"
    )
    
    print("✅ Model exported to 'hf_export/' directory")
    print("📝 Model card created")
    
    # Optional: Upload to HF Hub (requires token)
    upload_choice = input("\n🤔 Do you want to upload to Hugging Face Hub? (y/n): ").lower()
    
    if upload_choice == 'y':
        hf_token = input("🔑 Enter your HF token (or press Enter if logged in): ").strip()
        hf_token = hf_token if hf_token else None
        
        repo_name = input(f"📦 Repository name [{repo_name}]: ").strip() or repo_name
        
        upload_to_huggingface(
            model_path=model_path,
            config_path=config_path,
            class_names=class_names,
            model_name=model_name,
            repo_name=repo_name,
            hf_token=hf_token
        )
    else:
        print("ℹ️  Model exported locally. You can manually upload the 'hf_export/' directory to HF Hub")
        print("📖 Instructions:")
        print("   1. Create a new model repository on https://huggingface.co/new")
        print("   2. Clone the repository locally")
        print("   3. Copy files from 'hf_export/' to your cloned repository")
        print("   4. Commit and push the changes")

if __name__ == "__main__":
    main()
