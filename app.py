from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import base64
import io
import os
from pathlib import Path
import datetime
import pandas as pd

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Add CORS support for cross-origin requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'simple_stroke_model.pth'
CLASS_LABELS = ["Hemorrhagic", "Ischaemic", "Normal"]
IMG_SIZE = 128

# Create uploads directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple CNN model (same as training)
class SimpleStrokeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleStrokeClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the trained model with enhanced error handling
device = torch.device("cpu")  # Force CPU for Render free tier
model = None
model_loaded = False

print(f"📍 Attempting to load model from: {MODEL_PATH}")
print(f"📁 Current working directory: {os.getcwd()}")
print(f"📋 Files in directory: {os.listdir('.')}")

try:
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model file found: {MODEL_PATH}")
        print(f"📏 Model file size: {os.path.getsize(MODEL_PATH)} bytes")
        
        # Initialize model
        model = SimpleStrokeClassifier(num_classes=3)
        
        # Load with minimal memory usage
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model_loaded = True
        
        print(f"✅ Model loaded successfully!")
        print(f"🎯 Model validation accuracy: {checkpoint.get('val_acc', 'Unknown')}")
    else:
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("⚠️ App will run in demo mode without predictions")
        model_loaded = False
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"🐛 Error type: {type(e).__name__}")
    print("⚠️ App will run in demo mode without predictions")
    model_loaded = False
    model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(image_file):
    """Preprocess uploaded image for model prediction"""
    try:
        # Read image
        image = Image.open(image_file)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor, None
    except Exception as e:
        return None, str(e)

def predict_stroke(image_tensor):
    """Make stroke classification prediction"""
    if not model_loaded:
        # Demo mode - return mock prediction
        print("🎭 Running in demo mode - returning mock prediction")
        import random
        predicted_idx = random.randint(0, 2)
        confidence = random.uniform(0.6, 0.95)
        
        # Generate realistic probabilities
        probs = [random.uniform(0.05, 0.3) for _ in range(3)]
        probs[predicted_idx] = confidence
        # Normalize
        total = sum(probs)
        probs = [p/total for p in probs]
        
        result = {
            'predicted_class': CLASS_LABELS[predicted_idx],
            'confidence': confidence,
            'probabilities': {
                CLASS_LABELS[i]: float(prob) for i, prob in enumerate(probs)
            },
            'demo_mode': True
        }
        return result, None
    
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all class probabilities
            probs = probabilities[0].cpu().numpy()
            
            result = {
                'predicted_class': CLASS_LABELS[predicted.item()],
                'confidence': confidence.item(),
                'probabilities': {
                    CLASS_LABELS[i]: float(prob) for i, prob in enumerate(probs)
                },
                'demo_mode': False
            }
            
            return result, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    print(f"🔥 Predict endpoint called - Method: {request.method}")
    print(f"📁 Files in request: {list(request.files.keys())}")
    print(f"🌐 Request headers: {dict(request.headers)}")
    
    # Allow demo mode even without model
    if not model_loaded:
        print("⚠️ Model not loaded - will use demo mode")
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'})
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'})
    
    try:
        print(f"✅ Processing image: {image_file.filename}")
        
        # Preprocess image
        image_tensor, error = preprocess_image(image_file)
        if error:
            print(f"❌ Image preprocessing error: {error}")
            return jsonify({'success': False, 'error': f'Image processing error: {error}'})
        
        print(f"✅ Image preprocessed successfully, tensor shape: {image_tensor.shape}")
        
        # Make prediction
        result, error = predict_stroke(image_tensor)
        if error:
            print(f"❌ Prediction error: {error}")
            return jsonify({'success': False, 'error': f'Prediction error: {error}'})
        
        print(f"✅ Prediction successful: {result['predicted_class']} ({result['confidence']:.3f})")
        
        # Save uploaded image (optional)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{image_file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.seek(0)  # Reset file pointer
        image_file.save(filepath)
        
        # Add metadata
        result['filename'] = filename
        result['timestamp'] = timestamp
        result['success'] = True
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test', methods=['GET', 'POST'])
def test():
    """Test endpoint for debugging"""
    return jsonify({
        'status': 'test endpoint working',
        'method': request.method,
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': str(device),
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    # This is for local development only
    # In production, Gunicorn will handle the server
    port = int(os.environ.get('PORT', 5001))
    print("🧠 Stroke Classification Web App")
    print("=" * 40)
    print(f"🔧 Model loaded: {'✅ Yes' if model_loaded else '❌ No'}")
    print(f"🖥️  Device: {device}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🌐 Starting Flask server on port {port}...")
    print(f"🔗 Open http://localhost:{port} in your browser")
    
    # Use production settings when deployed
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
