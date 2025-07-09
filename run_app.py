#!/usr/bin/env python3
"""
Quick setup and run script for Workforce Distribution AI
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        sys.exit(1)

def check_model_files():
    """Check if all required model files exist"""
    required_files = [
        "model.pkl",
        "salary_predictor.pkl", 
        "role_classifier.pkl",
        "role_encoder.pkl",
        "imputer.pkl",
        "Employee.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n🔧 Run 'python train_model.py' to generate model files")
        return False
    
    print("✅ All required files found")
    return True

def train_models():
    """Train the models if they don't exist"""
    print("🤖 Training models...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Models trained successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to train models")
        sys.exit(1)

def run_streamlit():
    """Run the Streamlit app"""
    print("🚀 Starting Streamlit app...")
    print("📱 Open your browser and go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped. Thanks for using Workforce Distribution AI!")

def main():
    """Main function"""
    print("🎯 Workforce Distribution AI - Quick Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Check for model files
    if not check_model_files():
        user_input = input("\n❓ Train models now? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            train_models()
        else:
            print("❌ Cannot run app without model files")
            sys.exit(1)
    
    # Run the app
    run_streamlit()

if __name__ == "__main__":
    main()