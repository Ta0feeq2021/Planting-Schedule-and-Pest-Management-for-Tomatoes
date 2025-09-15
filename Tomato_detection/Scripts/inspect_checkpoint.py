import torch
import os

def inspect_checkpoint():
    """Inspect the contents of the saved model checkpoint"""
    
    # Try different possible model paths
    possible_paths = [
        "models/vit_tomato_model_best.pth",
        "models/vit_tomato_model.pth",
        "models/pest_model.pth"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No model file found! Looking for .pth files in current directory...")
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_files:
            print(f"Found .pth files: {pth_files}")
            model_path = pth_files[0]
            print(f"Using: {model_path}")
        else:
            print("No .pth files found in current directory")
            return
    
    print(f"📁 Loading checkpoint from: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"📊 Checkpoint contents:")
        print("-" * 50)
        
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor of shape {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}: Dictionary with {len(value)} items")
                    if len(value) < 20:  # Only show contents if not too many items
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")
        else:
            print(f"Checkpoint is not a dictionary, it's a {type(checkpoint).__name__}")
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

if __name__ == "__main__":
    inspect_checkpoint()
