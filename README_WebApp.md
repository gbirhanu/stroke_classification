# 🧠 Stroke Classification Web Application

A lightweight Flask web application for demonstrating stroke classification using a trained PyTorch model.

## 📋 Overview

This web application provides an intuitive interface to:
- Upload brain scan images (PNG, JPG, JPEG)
- Classify strokes into three categories: **Hemorrhagic**, **Ischaemic**, or **Normal**
- Display prediction confidence and detailed class probabilities
- Demonstrate your AI model to colleagues with a professional interface

## 🚀 Quick Start

### 1. Train the Model First
Before running the web app, you need to train the model:
```bash
# Open and run the Jupyter notebook
jupyter notebook stroke_training_simple.ipynb
```
- Run all cells to train the model (~10 minutes on laptop)
- This creates `simple_stroke_model.pth` file

### 2. Run the Web Application
Use the automated startup script:
```bash
python run_app.py
```

Or install dependencies manually and run:
```bash
# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

### 3. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure

```
stroke_multiclass/
├── app.py                           # Main Flask application
├── templates/
│   └── index.html                   # Web interface template
├── requirements.txt                 # Python dependencies
├── run_app.py                      # Startup script with checks
├── simple_stroke_model.pth         # Trained model (created after training)
├── uploads/                        # Uploaded images storage
├── stroke_training_simple.ipynb    # Training notebook
├── stroke_data_preprocessing.ipynb # Data preprocessing notebook
└── README_WebApp.md               # This file
```

## 🎯 Features

### User Interface
- **Drag & Drop Upload**: Easily upload brain scan images
- **Real-time Preview**: See uploaded images before analysis
- **Professional Design**: Clean, medical-grade interface
- **Responsive Layout**: Works on desktop and mobile

### AI Predictions
- **Multi-class Classification**: Hemorrhagic, Ischaemic, Normal
- **Confidence Scores**: See prediction certainty levels
- **Detailed Probabilities**: View all class probabilities
- **Visual Indicators**: Color-coded results for quick interpretation

### Technical Features
- **Model Validation**: Automatic checks for trained model
- **Error Handling**: Graceful handling of invalid inputs
- **Health Monitoring**: `/health` endpoint for status checks
- **File Management**: Automatic cleanup and organization

## 🔧 Configuration

### Supported Image Formats
- PNG, JPG, JPEG
- Maximum file size: 16MB
- Automatic conversion to grayscale

### Model Specifications
- **Input Size**: 128x128 grayscale images
- **Architecture**: Simple CNN (4-layer convolutional + 2-layer classifier)
- **Classes**: 3 (Hemorrhagic, Ischaemic, Normal)
- **Framework**: PyTorch

## 📊 Usage Examples

### For Demo/Presentation
1. Train the model using provided notebooks
2. Start the web app: `python run_app.py`
3. Open browser to `http://localhost:5000`
4. Upload sample brain scan images
5. Show real-time classification results

### For Development/Testing
```python
# Test the prediction endpoint
import requests

# Health check
response = requests.get('http://localhost:5000/health')
print(response.json())

# Upload image for prediction
files = {'image': open('brain_scan.png', 'rb')}
response = requests.post('http://localhost:5000/predict', files=files)
print(response.json())
```

## 🛠️ Troubleshooting

### Common Issues

**1. Model Not Found Error**
```
❌ Model not found: simple_stroke_model.pth
```
**Solution**: Run the training notebook first to create the model file.

**2. Dependency Errors**
```
❌ flask - MISSING
```
**Solution**: Install requirements: `pip install -r requirements.txt`

**3. Port Already in Use**
```
OSError: [Errno 48] Address already in use
```
**Solution**: Kill existing process or change port in `app.py`

**4. Upload Errors**
```
Image processing error: cannot identify image file
```
**Solution**: Ensure uploaded file is a valid image (PNG, JPG, JPEG)

### Performance Tips
- Use grayscale images for faster processing
- Resize large images before upload for better performance  
- Close browser tabs when not in use to free memory

## 🔒 Security Notes

- This is a **demo application** - not production ready
- No authentication or user management
- Uploaded files are stored locally in `uploads/` folder
- Add proper security measures for production deployment

## 📝 API Endpoints

### `GET /`
Main web interface

### `POST /predict`
Upload image for stroke classification
- **Input**: Form data with 'image' file
- **Output**: JSON with prediction results

### `GET /health`
Health check and status information
- **Output**: JSON with server status

### `GET /uploads/<filename>`
Serve uploaded image files

## 🎨 Customization

### Modify Class Labels
Edit `CLASS_LABELS` in `app.py`:
```python
CLASS_LABELS = ["Custom1", "Custom2", "Custom3"]
```

### Change Model Path
Update `MODEL_PATH` in `app.py`:
```python
MODEL_PATH = 'path/to/your/model.pth'
```

### Customize Interface
Edit `templates/index.html` to modify:
- Colors and styling
- Layout and components
- Text and labels

## 📚 Next Steps

1. **Improve Model**: Collect more data, try different architectures
2. **Add Features**: Batch processing, result history, export functionality
3. **Deploy**: Use Docker, cloud platforms, or dedicated servers
4. **Security**: Add authentication, input validation, rate limiting
5. **Monitoring**: Add logging, metrics, performance tracking

## 🤝 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the training notebook for model-related issues
3. Verify all dependencies are correctly installed
4. Test with different image formats/sizes

---

**Disclaimer**: This application is for research and demonstration purposes only. Always consult qualified medical professionals for actual medical diagnosis and treatment decisions.