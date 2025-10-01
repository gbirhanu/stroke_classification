#!/usr/bin/env python3
"""
Startup script for Stroke Classification Web Application

This script checks dependencies, validates the model, and starts the Flask web server.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'torch', 'torchvision', 'PIL', 'numpy', 'cv2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install missing requirements"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        print("📦 Installing requirements from requirements.txt...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    else:
        print("⚠️ requirements.txt not found")
        return False

def check_model():
    """Check if the trained model exists"""
    model_path = Path(__file__).parent / "simple_stroke_model.pth"
    
    if model_path.exists():
        print(f"✅ Model found: {model_path}")
        return True
    else:
        print(f"❌ Model not found: {model_path}")
        print("⚠️  Please run the training notebook first to create the model")
        return False

def main():
    print("🧠 Stroke Classification Web App - Startup")
    print("=" * 50)
    
    # Check current directory
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing packages: {missing}")
        choice = input("Do you want to install missing packages? (y/N): ").strip().lower()
        
        if choice == 'y':
            if not install_requirements():
                print("❌ Failed to install requirements. Exiting.")
                return 1
        else:
            print("❌ Cannot proceed without required packages. Exiting.")
            return 1
    
    # Check model
    print("\n🔍 Checking trained model...")
    model_exists = check_model()
    
    if not model_exists:
        print("\n💡 To create the model:")
        print("   1. Open and run 'stroke_training_simple.ipynb'")
        print("   2. Train the model (takes ~10 minutes)")
        print("   3. Come back and run this script again")
        
        choice = input("\nDo you want to continue without model (web app will show warning)? (y/N): ").strip().lower()
        if choice != 'y':
            print("👋 Exiting. Please train the model first.")
            return 1
    
    # Start Flask app
    print("\n🚀 Starting Flask web application...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())