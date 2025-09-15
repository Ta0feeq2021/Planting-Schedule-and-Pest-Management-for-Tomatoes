from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
from flask_pymongo import PyMongo
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import io
import json
from markupsafe import Markup
from user.utils.pestid import pest_name
from datetime import datetime
import torch
import timm
import torch.nn as nn
from torchvision import transforms
import os
from werkzeug.utils import secure_filename
from bson import ObjectId
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from flask_cors import CORS
import json
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])
app.config['MONGO_URI'] = "mongodb://localhost:27017/pest"
mongo = PyMongo(app)
# CORS(app, origins=['http://localhost:3000'])

UPLOAD_FOLDER = 'user/static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

username = "officer"
password = "officer123"
app.secret_key = 'pest_detection_key'


# image prediction ..........................................................................................................................
# Define your class labels as a list
pest_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

# Load the model
pest_model_path = 'models/vit_tomato_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InsectModel(nn.Module):
    def __init__(self, num_classes, device='cpu'):
        super(InsectModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)

        pest_model_path = 'models/pest_model.pth'

        huggingface_model_path =r'C:\Users\taofe\.cache\torch\hub\checkpoints\vit_base_patch16_224-augreg_in1k.pth'
        
        # if os.path.exists(pest_model_path):
        #     print(f"Loading pest model from {pest_model_path}")
        #     self.model.load_state_dict(torch.load(pest_model_path, map_location=self.device))
        # elif os.path.exists(huggingface_model_path):
        #     print(f"Loading HuggingFace model from {huggingface_model_path}")
        #     self.model.load_state_dict(torch.load(huggingface_model_path, map_location=self.device))
        # else:
        #     print("Warning: No model file found! Using untrained model.")
        # # self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        if os.path.exists(pest_model_path):
            print(f"Loading pest model from {pest_model_path}")
            state_dict = torch.load(pest_model_path, map_location=self.device)

            # 🔧 Strip 'model.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "") if k.startswith("model.") else k
                new_state_dict[new_key] = v

            self.model.load_state_dict(new_state_dict)

        elif os.path.exists(huggingface_model_path):
            print(f"Loading HuggingFace model from {huggingface_model_path}")
            self.model.load_state_dict(torch.load(huggingface_model_path, map_location=self.device))

        else:
            print("⚠️ Warning: No model file found! Using untrained model.")

    
    def forward(self, image):
        return self.model(image)
    

def train_transform():
    return A.Compose([
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.RandomBrightnessContrast(),
        A.Resize(224, 224),
        ToTensorV2()])

def valid_transform():
    return A.Compose([
        A.Resize(224,224),
        ToTensorV2()])


model = InsectModel(num_classes=40, device=device)
model.load_state_dict(torch.load(pest_model_path, map_location=device))
model.to(device)
model.eval()

def predict_image(img, model=model, classes=pest_classes):
    transform = valid_transform()
    image = Image.open(io.BytesIO(img)).convert("RGB")  
    transformed = transform(image=np.array(image))
    img_t = transformed['image']
    # Convert PyTorch tensor to NumPy array
    img_np = img_t.numpy().astype(np.float32)  # Convert image to float32
    img_np /= 255.0  # Normalize to [0, 1] range
    # Convert NumPy array back to PyTorch tensor
    img_u = torch.from_numpy(img_np).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_u)
        _, preds = torch.max(outputs, dim=1)
        prediction = classes[preds.item()] 
    return prediction

    print(" predict_image function was called.")




# set path to each webpages..................................................................................................
@app.route('/')
def home():
    title = 'Pest - Home'
    return render_template('index.html', title=title)

@ app.route('/predict')
def predict():
    title = 'Prediction'
    return render_template('home.html', title=title)

@ app.route('/user_query')
def user_query():
    title = 'User Queries'
    return render_template('user_query.html', title=title)

@app.route('/officer_login', methods=['GET'])
def officer_login():
    title = 'Officer - Login'
    return render_template('officer_login.html', title=title)

@ app.route('/officer')
def officer():
    title = 'Officer - Home'
    return render_template('officer.html', title=title)


# prediction.................................................................................................................
# @app.route('/predict', methods=['POST','GET'])
# def make_prediction():
#     title = 'Pest - Detection'
#     pest_collection = mongo.db.pest_details
    
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files.get('file')
#         if not file:
#             return render_template('home.html', title=title)
#         try:
#             img = file.read()
#             prediction = predict_image(img)  # Get the prediction
#             name=pest_name.get(prediction)
#             pest = pest_collection.find_one({'name': name })

#             return render_template('result.html', prediction=prediction, pest=pest, title=title)
#         except Exception as e:
#             print(f"Prediction error: {str(e)}")
#             pass  # Handle the exception or error accordingly

#     return render_template('home.html', title=title)

@app.route('/predict', methods=['POST','GET'])
def make_prediction():
    title = 'Pest - Detection'
    pest_collection = mongo.db.pest_details
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file selected'}), 400
            
        try:
            img = file.read()
            prediction = predict_image(img)
            name = pest_name.get(prediction)
            pest = pest_collection.find_one({'name': name})
            
            if pest:
                # Convert MongoDB document to JSON-serializable format
                pest_data = {
                    'name': pest.get('name', ''),
                    'description': pest.get('description', ''),
                    'prevention': pest.get('prevention', '').split('\n') if pest.get('prevention') else [],
                    'pesticides': pest.get('pesticides', '').split('\n') if pest.get('pesticides') else [],
                    'pest_image': pest.get('pest_image', [])
                }
                
                # Calculate confidence score (implement based on your model)
                confidence = 85.0  # Replace with actual confidence from your model
                
                result = {
                    'pestName': name,
                    'confidence': confidence,
                    'description': pest_data['description'],
                    'prevention': pest_data['prevention'],
                    'pesticides': pest_data['pesticides'],
                    'images': pest_data['pest_image']
                }
                
                # Log prediction to MongoDB
                try:
                    mongo.db.predictions.insert_one({
                        'timestamp': datetime.now(),
                        'pestName': name,
                        'confidence': confidence,
                        'prediction_id': prediction,
                        'file_size': len(img)
                    })
                except Exception as log_error:
                    print(f"Logging error: {log_error}")
                
                return jsonify(result)
            else:
                return jsonify({'error': 'Pest not found in database'}), 404
                
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # For GET requests, return the HTML form (existing behavior)
    return render_template('home.html', title=title)

@app.route('/api/predict_with_confidence', methods=['POST'])
def predict_with_confidence():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file selected'}), 400
            
        img = file.read()
        
        # Get prediction with confidence
        prediction = predict_image(img)
        confidence = 85.0  # Replace with actual confidence calculation
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'pest_name': pest_name.get(prediction, 'Unknown')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/view_query')
def display_data():
    data = list(mongo.db.user_query.find().sort('timestamp', -1))
    for index, query in enumerate(data, start=1):
        query['serial'] = index
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the current date and time as a string
    formatted_current_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # Format the timestamp in your query data with a space between date and time
    for query in data:
        query['date'] = query['timestamp'].strftime('%Y-%m-%d')
        query['time'] = query['timestamp'].strftime('%H:%M:%S')

    return render_template('view_query.html', title = 'User Queries', data=data, current_datetime=formatted_current_datetime)

@app.route('/view_query')
def view_query():
    
    title = 'User Queries'
    return render_template('view_query.html', title=title)



# officer login page validation..............................................................................................
@app.route('/officer_login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname == username and pwd == password:
            # Successful login - redirect to officer page
            return redirect(url_for('officer'))
        else:
            # Incorrect credentials - redirect back to login page with a flash message
            flash('Invalid username or password. Please try again.', 'error')
            return redirect(url_for('login'))

    return render_template('officer_login.html', title='Officer - Login')


# pest details...............................................................................................................
@app.route('/officer')
def pest_info():
	with open("dictionary.json", "r") as f:
	# Read the dictionary file and convert it to a JSON object
		pest_info = list[json.load(f)]
	return render_template('officer.html', pest_info=pest_info)

@app.route('/detailed_view/<pest_id>')
def detailed_view(pest_id):
    # pest = pest_dic.get(pest_id)
    # return render_template('detailed_pestview.html', pest=pest, title="Pest - Description")
    id=pest_id
    pest_collection = mongo.db.pest_details
    name=pest_name.get(pest_id)
    try:
        pest = pest_collection.find_one({'name': name })
        if pest:
            return render_template('detailed_pestview.html', pest=pest, pest_id=id, title="Pest - Description")
        
        else:
            return "Pest not found"
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/detailed_view/<pest_id>', methods=['GET','POST'])        
def edit_pest(pest_id):
    id = pest_id
    pest_collection = mongo.db.pest_details
    name = pest_name.get(pest_id)
    query = {'name': name}
    
    if name:
        if request.method == 'POST':
            # Handle text information update
            description = request.form.get('description')
            prevention = request.form.get('prevention')
            pesticides = request.form.get('pesticides')
            
            pest_collection.update_one(query, {'$set': {
                'description': description,
                'prevention': prevention,
                'pesticides': pesticides
            }})

            # Handle delete images
            deleted_images = request.form.getlist('deleted_images')
            for image_path in deleted_images:
                pest_collection.update_one(query, {'$pull': {'pest_image': image_path}})


            # Handle image upload
            if 'image' in request.files:
                file = request.files['image']
                if file:
                    filename = secure_filename(file.filename)
                    image_name = "pesticides/" + filename
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
                    file.save(file_path)
                    pest_collection.update_one(query, {'$push': {'pest_image': image_name}})
        
            return redirect(url_for('edit_pest', pest_id=pest_id))
    else:
        pest_data = pest_collection.find_one(query)
        return render_template('detailed_pestview.html', pest=pest_data, pest_id=id)


# submit user query .................................................................................................................
@app.route('/submit_query', methods=['POST'])
def submit_query():
    data = {
        'name': request.form.get('Name'),
        'email': request.form.get('Email'),
        'message': request.form.get('Message'),
        'timestamp': datetime.now()
    }
    mongo.db.user_query.insert_one(data)
    return render_template('user_query.html')

@app.route('/')
def logout():
	# Clear session data
    session.clear()
    # Redirect to the login page or any other page after logout
    return redirect(url_for('home'))



