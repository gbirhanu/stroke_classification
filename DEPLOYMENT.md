# Deployment Guide for Stroke Classification App

## Render Deployment

### Quick Setup
1. Connect your GitHub repository to Render
2. Set the following configuration in Render:

**Build Command:**
```
pip install -r requirements.txt
```

**Start Command:**
```
gunicorn --config gunicorn_config.py app:app
```

### Environment Variables (Optional)
- `FLASK_ENV=production` (automatically set in production)
- `PORT` (automatically set by Render)

### File Structure
```
stroke_multiclass/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── simple_stroke_model.pth   # Trained model file
├── templates/
│   └── index.html           # Web interface
├── uploads/                 # File uploads directory
├── gunicorn_config.py       # Production server config
├── start.sh                 # Alternative startup script
└── .python-version         # Python 3.11
```

### Production URL
Your app will be available at: `https://your-app-name.onrender.com`

### Health Check
Visit `/health` endpoint to check if the model loaded successfully:
`https://your-app-name.onrender.com/health`

## Local Development

Run locally:
```bash
python app.py
```
or
```bash
python run_app.py
```

## Troubleshooting

### Common Issues:
1. **Model not loading**: The app will still work but show a warning
2. **Port issues**: Render automatically assigns PORT environment variable
3. **Memory limits**: Free tier has 512MB RAM limit

### Logs:
Check Render logs for any deployment issues.