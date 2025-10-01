#!/bin/bash

# Production startup script for Render deployment
echo "🚀 Starting Stroke Classification App on Render..."

# Get the port from environment (Render sets this)
export PORT=${PORT:-10000}
echo "📡 Port: $PORT"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if model exists
if [ -f "simple_stroke_model.pth" ]; then
    echo "✅ Model file found"
else
    echo "⚠️  Model file not found - app will show warning"
fi

# Start with Gunicorn
echo "🌐 Starting Gunicorn server..."
exec gunicorn --config gunicorn_config.py app:app