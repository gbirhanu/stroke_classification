#!/usr/bin/env python3
"""
Startup diagnostic script for Render deployment
"""
import os
import sys
import traceback
import psutil

def check_environment():
    """Check environment and resources"""
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"PATH: {os.environ.get('PATH', 'Not set')[:200]}...")
    print(f"PORT: {os.environ.get('PORT', 'Not set')}")
    
    # Memory check
    try:
        memory = psutil.virtual_memory()
        print(f"Available memory: {memory.available / (1024*1024):.1f} MB")
        print(f"Total memory: {memory.total / (1024*1024):.1f} MB")
        print(f"Memory usage: {memory.percent}%")
    except:
        print("Could not check memory")
    
    print("\n📁 FILES IN DIRECTORY:")
    for item in sorted(os.listdir('.')):
        if os.path.isfile(item):
            size = os.path.getsize(item)
            print(f"  {item} ({size:,} bytes)")
        else:
            print(f"  {item}/ (directory)")

def check_dependencies():
    """Check if all dependencies can be imported"""
    print("\n📦 DEPENDENCY CHECK")
    print("=" * 50)
    
    deps = [
        'flask', 'torch', 'torchvision', 'PIL', 'numpy', 
        'cv2', 'pandas', 'gunicorn'
    ]
    
    for dep in deps:
        try:
            if dep == 'PIL':
                import PIL
                print(f"✅ {dep} - OK")
            elif dep == 'cv2':
                import cv2
                print(f"✅ {dep} - OK")
            else:
                __import__(dep)
                print(f"✅ {dep} - OK")
        except Exception as e:
            print(f"❌ {dep} - ERROR: {e}")

def test_app_import():
    """Test if the Flask app can be imported"""
    print("\n🧪 APP IMPORT TEST")
    print("=" * 50)
    
    try:
        from app import app, model_loaded
        print("✅ Flask app imported successfully")
        print(f"✅ Model loaded: {model_loaded}")
        return True
    except Exception as e:
        print(f"❌ Failed to import app: {e}")
        print("Error details:")
        traceback.print_exc()
        return False

def main():
    print("🚀 RENDER DEPLOYMENT DIAGNOSTIC")
    print("=" * 50)
    
    check_environment()
    check_dependencies()
    success = test_app_import()
    
    print(f"\n📊 SUMMARY")
    print("=" * 50)
    if success:
        print("✅ App should start successfully")
        return 0
    else:
        print("❌ App will likely fail to start")
        return 1

if __name__ == "__main__":
    sys.exit(main())