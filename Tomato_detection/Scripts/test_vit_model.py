import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os
import json

def test_vit_model():
    """Test the trained ViT model with sample images"""
    
    # Configuration - try multiple possible locations
    possible_paths = [
        "models/vit_tomato_model_best.pth",
        "models/vit_tomato_model.pth",
        "models/pest_model.pth"
    ]
    
    MODEL_PATH = None
    for path in possible_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            break
    
    if MODEL_PATH is None:
        print("❌ Model not found in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease locate your .pth model file and update the path.")
        return
    
    # Load the model
    print(f"🔄 Loading trained model from {MODEL_PATH}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        print("✅ Checkpoint loaded successfully!")
        
        if 'class_to_idx' in checkpoint:
            class_to_idx = checkpoint['class_to_idx']
            num_classes = len(class_to_idx)
            # Convert class_to_idx to class_names (reverse the mapping)
            class_names = [None] * num_classes
            for class_name, idx in class_to_idx.items():
                class_names[idx] = class_name
        else:
            print("❌ No class information found in checkpoint")
            return
        
        val_acc = checkpoint.get('val_acc', 'Unknown')
        print(f"📊 Model validation accuracy: {val_acc:.2f}%" if isinstance(val_acc, (int, float)) else f"📊 Model validation accuracy: {val_acc}")
        print(f"🏷️ Number of classes: {num_classes}")
        print(f"🐛 Pest classes: {', '.join(class_names)}")
        
        # Initialize model architecture
        print("🔄 Initializing model architecture...")
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print("✅ Model loaded and ready for testing!")
        
        # Initialize image processor
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Test with sample images from dataset
        test_image_path = "classification_dataset_for_vit/test"
        if os.path.exists(test_image_path):
            print(f"\n🧪 Testing with sample images from {test_image_path}")
            
            # Get first few images from each class
            tested_classes = 0
            for class_name in class_names:
                if tested_classes >= 3:  # Test max 3 classes
                    break
                    
                class_path = os.path.join(test_image_path, class_name)
                if os.path.exists(class_path):
                    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        image_path = os.path.join(class_path, images[0])
                        test_single_image(image_path, model, processor, class_names, device)
                        tested_classes += 1
        else:
            print(f"❌ Test dataset not found at {test_image_path}")
            print("To test the model, you can:")
            print("1. Place test images in the classification_dataset_for_vit/test/ folder")
            print("2. Or provide a specific image path to test")
            
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return

def test_single_image(image_path, model, processor, class_names, device):
    """Test model on a single image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        predicted_class = class_names[predicted_class_idx]
        actual_class = os.path.basename(os.path.dirname(image_path))
        
        print(f"\n📸 Image: {os.path.basename(image_path)}")
        print(f"🎯 Predicted: {predicted_class} ({confidence:.2%} confidence)")
        print(f"✅ Actual: {actual_class}")
        print(f"{'✅ CORRECT' if predicted_class == actual_class else '❌ INCORRECT'}")
        
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        print("🔝 Top 3 predictions:")
        for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            print(f"   {i+1}. {class_names[idx.item()]}: {prob.item():.2%}")
        
    except Exception as e:
        print(f"❌ Error testing image {image_path}: {str(e)}")

if __name__ == "__main__":
    test_vit_model()
