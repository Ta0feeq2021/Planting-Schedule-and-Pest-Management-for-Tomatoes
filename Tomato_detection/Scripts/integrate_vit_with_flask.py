"""
Integration script to connect trained ViT model with Flask backend
"""

import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import json
import numpy as np
from torchvision import transforms
import io
import os

class ViTTomatoPredictor:
    """ViT model predictor for Flask integration"""
    
    def __init__(self, model_path='models/vit_tomato_model_best.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not os.path.exists(model_path):
            # Try alternative paths
            alternative_paths = [
                'models/vit_tomato_model_final.pth',
                'vit_tomato_model_best.pth',
                'vit_tomato_model_final.pth'
            ]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Model not found at {model_path} or alternative paths")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        self.num_classes = len(self.class_names)
        
        # Create class mappings
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        
        # Load model architecture
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ ViT model loaded successfully from {model_path}!")
        print(f"📱 Device: {self.device}")
        print(f"🏷️  Classes ({self.num_classes}): {self.class_names}")
    
    def predict(self, image_bytes):
        """
        Predict pest from image bytes
        Returns: (class_name, confidence, all_probabilities)
        """
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class_idx].item()
                
                predicted_class = self.class_names[predicted_class_idx]
                confidence_score = confidence * 100
                all_probs = probabilities.cpu().numpy()[0]
            
            return predicted_class, confidence_score, all_probs
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, None
    
    def get_top_predictions(self, image_bytes, top_k=3):
        """Get top-k predictions"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                results = []
                for i in range(top_k):
                    class_idx = top_indices[0][i].item()
                    class_name = self.class_names[class_idx]
                    confidence = top_probs[0][i].item() * 100
                    results.append((class_name, confidence))
                
                return results
                
        except Exception as e:
            print(f"Top predictions error: {e}")
            return []

flask_integration_code = '''
# Add this to your Flask app (flask-backend/user.py)

from scripts.integrate_vit_with_flask import ViTTomatoPredictor
import json
from datetime import datetime

# Initialize ViT predictor (do this once when Flask starts)
try:
    vit_predictor = ViTTomatoPredictor(model_path='models/vit_tomato_model_best.pth')
    print("✅ ViT model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load ViT model: {e}")
    vit_predictor = None

try:
    with open('flask-backend/tomato_pest_database.json', 'r') as f:
        tomato_pest_db = json.load(f)
except FileNotFoundError:
    print("⚠️ tomato_pest_database.json not found, using basic pest info")
    tomato_pest_db = {}

@app.route('/predict', methods=['POST','GET'])
def predict_with_vit():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file selected'}), 400
            
        try:
            # Read image bytes
            img_bytes = file.read()
            
            if vit_predictor is None:
                return jsonify({'error': 'ViT model not loaded'}), 500
            
            # Get prediction from ViT model
            predicted_class, confidence, all_probs = vit_predictor.predict(img_bytes)
            
            if predicted_class is None:
                return jsonify({'error': 'Prediction failed'}), 500
            
            # Get pest information from database
            pest_info = tomato_pest_db.get(predicted_class, {})
            
            # Clean up class name for display
            display_name = predicted_class.replace('Tomato___', '').replace('_', ' ').title()
            
            # Get top 3 predictions for additional context
            top_predictions = vit_predictor.get_top_predictions(img_bytes, top_k=3)
            
            result = {
                'pestName': pest_info.get('name', display_name),
                'confidence': round(confidence, 2),
                'description': pest_info.get('description', f'Detected {display_name} with {confidence:.1f}% confidence'),
                'symptoms': pest_info.get('symptoms', ['Symptoms data not available']),
                'prevention': pest_info.get('prevention', ['Prevention data not available']),
                'pesticides': pest_info.get('treatment', ['Treatment data not available']),
                'severity': pest_info.get('severity', 'Medium'),
                'scientific_name': pest_info.get('scientific_name', ''),
                'optimal_conditions': pest_info.get('optimal_conditions', 'Varies by pest type'),
                'top_predictions': [
                    {
                        'name': pred[0].replace('Tomato___', '').replace('_', ' ').title(),
                        'confidence': round(pred[1], 2)
                    } for pred in top_predictions
                ],
                'model_type': 'ViT (Vision Transformer)',
                'images': [f'/static/pest_images/{predicted_class.lower()}.jpg']
            }
            
            # Log prediction to MongoDB
            try:
                mongo.db.predictions.insert_one({
                    'timestamp': datetime.now(),
                    'predicted_class': predicted_class,
                    'display_name': display_name,
                    'confidence': confidence,
                    'file_size': len(img_bytes),
                    'model_type': 'ViT',
                    'top_predictions': top_predictions
                })
            except Exception as log_error:
                print(f"MongoDB logging error: {log_error}")
            
            return jsonify(result)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # For GET requests, return the HTML form
    return render_template('home.html')

# Add endpoint for model info
@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    if vit_predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'Vision Transformer (ViT)',
        'classes': vit_predictor.class_names,
        'num_classes': vit_predictor.num_classes,
        'device': vit_predictor.device,
        'input_size': '224x224',
        'model_name': 'google/vit-base-patch16-224-in21k',
        'training_accuracy': '98.72%',
        'validation_accuracy': '97.21%'
    })
'''

def create_flask_integration_file():
    """Create the Flask integration file"""
    with open('flask_vit_integration.py', 'w') as f:
        f.write(flask_integration_code)
    
    print("📁 Created flask_vit_integration.py")
    print("💡 Copy the code from this file to your Flask app (flask-backend/user.py)")

def main():
    print("🔧 ViT-Flask Integration Setup")
    print("=" * 40)
    
    model_paths = [
        'models/vit_tomato_model_best.pth',
        'models/vit_tomato_model_final.pth',
        'vit_tomato_model_best.pth',
        'vit_tomato_model_final.pth'
    ]
    
    model_found = None
    for path in model_paths:
        if os.path.exists(path):
            model_found = path
            break
    
    if not model_found:
        print("❌ No trained model found!")
        print("🔍 Searched for:")
        for path in model_paths:
            print(f"  - {path}")
        print("\n💡 Please run the training script first:")
        print("   python scripts/train_vit_tomato_model.py")
        return
    
    # Test model loading
    try:
        predictor = ViTTomatoPredictor(model_path=model_found)
        print(f"✅ Model integration test successful using {model_found}!")
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        return
    
    # Create integration file
    create_flask_integration_file()
    
    print("\n🚀 Integration setup completed!")
    print("📋 Next steps:")
    print("1. Copy the code from flask_vit_integration.py to your Flask app")
    print("2. Make sure transformers and torch are installed in your Flask environment")
    print("3. Restart your Flask app")
    print("4. Test with tomato pest images")

if __name__ == "__main__":
    main()
