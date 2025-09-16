
            # alternative_paths = [
            #     '../models/vit_tomato_model_final.pth',
            #     '../models/vit_tomato_model.pth',
            #     '../vit_tomato_model_best.pth',
            #     '../vit_tomato_model_final.pth',
            #     'models/vit_tomato_model_best.pth',
            #     'vit_tomato_model_best.pth',
            #     '../models/vit_tomato_model_best.pth'
            #     "../models/vit_tomato_model_best.pth",
            #     "../models/vit_tomato_model.pth",
            #     "../models/pest_model.pth"
                
            # ]



"""
Complete Flask backend with ViT model integration for tomato pest detection
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from pymongo import MongoClient
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import json
import numpy as np
import io
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import requests
from pathlib import Path

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])



# Load models globally at startup (not per request)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit_model = torch.load('models/vit_tomato_model.pth', map_location=device)
pest_model = torch.load('models/pest_model.pth', map_location=device)

vit_model.eval()  # Set to evaluation mode
pest_model.eval()

db_connected = False

try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['tomato_pest_db']
    pest_collection = db.pest_details
    predictions_collection = db.predictions
    client.admin.command('ping')
    db_connected = True
    print("✅ MongoDB connected successfully")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    db = None

class ViTTomatoPredictor:
    """ViT model predictor for Flask integration"""
    
    def __init__(self, model_path='../models/vit_tomato_model_best.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not os.path.exists(model_path):
            alternative_paths = [
                '../models/vit_tomato_model_final.pth',
                 '../models/vit_tomato_model.pth',
                 '../vit_tomato_model_best.pth',
                 '../vit_tomato_model_final.pth',
                 'models/vit_tomato_model_best.pth',
                 'vit_tomato_model_best.pth',
                 '../models/vit_tomato_model_best.pth'
                 "../models/vit_tomato_model_best.pth",
                 "../models/vit_tomato_model.pth",
                 "../models/pest_model.pth"
            ]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Model not found at {model_path} or alternative paths")
        
        print(f"🔄 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract class information from checkpoint
        if 'class_to_idx' in checkpoint:
            self.class_to_idx = checkpoint['class_to_idx']
            self.class_names = list(self.class_to_idx.keys())
        else:
            raise KeyError("Checkpoint missing class_to_idx information")
        
        self.num_classes = len(self.class_names)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
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
        print(f"🏷️ Classes ({self.num_classes}): {self.class_names}")
    
    def predict(self, image_bytes):
        """Predict pest from image bytes"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
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

try:
    vit_predictor = ViTTomatoPredictor()
    print("✅ ViT model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load ViT model: {e}")
    vit_predictor = None

try:
    with open('tomato_pest_database.json', 'r') as f:
        tomato_pest_db = json.load(f)
except FileNotFoundError:
    print("⚠️ tomato_pest_database.json not found, using basic pest info")
    tomato_pest_db = {
        'Aphid': {
            'name': 'Aphid',
            'description': 'Small, soft-bodied insects that feed on plant sap',
            'symptoms': ['Yellowing leaves', 'Stunted growth', 'Honeydew secretion'],
            'prevention': ['Use reflective mulch', 'Encourage beneficial insects', 'Regular monitoring'],
            'treatment': ['Insecticidal soap', 'Neem oil', 'Systemic insecticides'],
            'severity': 'Medium'
        },
        'Leafhopper': {
            'name': 'Leafhopper',
            'description': 'Small jumping insects that feed on plant juices',
            'symptoms': ['White stippling on leaves', 'Leaf curling', 'Reduced plant vigor'],
            'prevention': ['Remove weeds', 'Use row covers', 'Maintain plant health'],
            'treatment': ['Pyrethrin sprays', 'Beneficial predators', 'Sticky traps'],
            'severity': 'Medium'
        },
        'Spider Mite': {
            'name': 'Spider Mite',
            'description': 'Tiny arachnids that cause stippling damage to leaves',
            'symptoms': ['Fine webbing', 'Yellow stippling', 'Leaf bronzing'],
            'prevention': ['Maintain humidity', 'Avoid over-fertilizing', 'Regular watering'],
            'treatment': ['Miticides', 'Predatory mites', 'Horticultural oils'],
            'severity': 'High'
        },
        'Spodoptera Larva': {
            'name': 'Spodoptera Larva',
            'description': 'Caterpillars that cause significant damage to tomato plants',
            'symptoms': ['Large holes in leaves', 'Fruit damage', 'Defoliation'],
            'prevention': ['Crop rotation', 'Pheromone traps', 'Early detection'],
            'treatment': ['Bt sprays', 'Chemical insecticides', 'Hand picking'],
            'severity': 'High'
        },
        'Spodoptera moth': {
            'name': 'Spodoptera moth',
            'description': 'Adult moths that lay eggs leading to larval damage',
            'symptoms': ['Egg masses on leaves', 'Adult moths present', 'Larval damage'],
            'prevention': ['Light traps', 'Pheromone traps', 'Crop monitoring'],
            'treatment': ['Targeted spraying', 'Biological control', 'Integrated pest management'],
            'severity': 'High'
        },
        'Stinkbug': {
            'name': 'Stinkbug',
            'description': 'Shield-shaped bugs that pierce and suck plant juices',
            'symptoms': ['Dimpled fruit', 'Wilting', 'Cloudy spot disease'],
            'prevention': ['Row covers', 'Trap crops', 'Garden sanitation'],
            'treatment': ['Pyrethroid insecticides', 'Beneficial predators', 'Physical removal'],
            'severity': 'Medium'
        },
        'Thrips': {
            'name': 'Thrips',
            'description': 'Tiny insects that rasp leaf surfaces and suck plant juices',
            'symptoms': ['Silver streaks on leaves', 'Black specks', 'Distorted growth'],
            'prevention': ['Blue sticky traps', 'Reflective mulch', 'Proper spacing'],
            'treatment': ['Insecticidal soap', 'Predatory mites', 'Systemic insecticides'],
            'severity': 'Medium'
        }
    }

@app.route('/')
def home():
    """Home page with upload form"""
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tomato Pest Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .result { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🍅 Tomato Pest Detection System</h1>
        <p>Upload an image of a tomato plant to detect pests using our AI model.</p>
        
        <div class="upload-form">
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <br><br>
                <button type="submit">Detect Pest</button>
            </form>
        </div>
        
        <h2>Model Information</h2>
        <p><strong>Model:</strong> Vision Transformer (ViT)</p>
        <p><strong>Classes:</strong> {{ num_classes }} African tomato pests</p>
        <p><strong>Accuracy:</strong> 97.21% validation accuracy</p>
    </body>
    </html>
    '''
    num_classes = len(vit_predictor.class_names) if vit_predictor else 0
    return render_template_string(html_template, num_classes=num_classes)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """Main prediction endpoint"""
    if request.method == 'GET':
        return home()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if vit_predictor is None:
        return jsonify({'error': 'ViT model not loaded'}), 500
    
    try:
        img_bytes = file.read()
        predicted_class, confidence, all_probs = vit_predictor.predict(img_bytes)
        
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        pest_info = tomato_pest_db.get(predicted_class, {})
        display_name = predicted_class.replace('_', ' ').title()
        
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
            'top_predictions': [
                {
                    'name': pred[0].replace('_', ' ').title(),
                    'confidence': round(pred[1], 2)
                } for pred in top_predictions
            ],
            'model_type': 'ViT (Vision Transformer)',
            'images': [f'/static/pest_images/{predicted_class.lower()}.jpg']
        }
        
        if db_connected and db is not None:
            try:
                predictions_collection.insert_one({
                    'timestamp': datetime.now(),
                    'predicted_class': predicted_class,
                    'display_name': display_name,
                    'confidence': confidence,
                    'file_size': len(img_bytes),
                    'model_type': 'ViT',
                    'top_predictions': top_predictions
                })
                print(f"✅ Prediction logged to MongoDB: {predicted_class} ({confidence:.2f}%)")
            except Exception as log_error:
                print(f"❌ MongoDB logging error: {log_error}")
        else:
            print("⚠️ MongoDB not connected, skipping database logging")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict_with_confidence', methods=['POST'])
def predict_with_confidence():
    """Simplified prediction endpoint with confidence"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file selected'}), 400
        
    if vit_predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        img_bytes = file.read()
        predicted_class, confidence, _ = vit_predictor.predict(img_bytes)
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'pest_name': predicted_class.replace('_', ' ').title()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get model information"""
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': vit_predictor is not None,
        'database_connected': db_connected,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Tomato Pest Detection Flask Server...")
    print("📱 Model classes:", vit_predictor.class_names if vit_predictor else "Model not loaded")
    print("🌐 Server will run on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

