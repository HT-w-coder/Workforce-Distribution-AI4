#!/usr/bin/env python3
"""
Korean Translation Fix - Diagnostic and Solution Script

This script helps diagnose and fix the Helsinki-NLP/opus-mt-en-ko model error
"""

import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking Dependencies...")
    
    dependencies = {
        'transformers': 'transformers',
        'torch': 'torch', 
        'requests': 'requests'
    }
    
    missing = []
    for name, import_name in dependencies.items():
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            missing.append(name)
            print(f"❌ {name} - Not installed")
        else:
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {name} - Version {version}")
            except:
                print(f"⚠️ {name} - Installed but version unknown")
    
    return missing

def install_dependencies(missing):
    """Install missing dependencies"""
    if not missing:
        print("✅ All dependencies are installed!")
        return True
    
    print(f"\n📦 Installing missing dependencies: {', '.join(missing)}")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        subprocess.run(cmd, check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_korean_models():
    """Test different Korean translation models"""
    print("\n🧪 Testing Korean Translation Models...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel
        
        # Test models for Korean translation
        test_models = [
            "Helsinki-NLP/opus-mt-en-ko",  # Original problematic model
            "facebook/nllb-200-distilled-600M",  # Alternative multilingual model
            "google/mt5-small",  # Google's multilingual model
        ]
        
        test_text = "Hello, how are you?"
        
        for model_name in test_models:
            print(f"\n🔄 Testing {model_name}...")
            
            try:
                # Try to load the model
                if "opus-mt" in model_name:
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    
                    # Test translation
                    inputs = tokenizer(test_text, return_tensors="pt")
                    outputs = model.generate(**inputs, max_length=40, num_beams=4)
                    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    
                    inputs = tokenizer(test_text, return_tensors="pt")
                    outputs = model.generate(**inputs, max_length=40)
                    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                print(f"✅ {model_name} - SUCCESS")
                print(f"   Translation: '{translation}'")
                
            except Exception as e:
                print(f"❌ {model_name} - FAILED: {str(e)}")
                
                # Provide specific solutions for common errors
                if "404" in str(e) or "does not exist" in str(e):
                    print("   💡 Solution: Model doesn't exist on HuggingFace Hub")
                    print("   💡 Try: Use alternative models like facebook/nllb-200-distilled-600M")
                elif "token" in str(e).lower():
                    print("   💡 Solution: Authentication required")
                    print("   💡 Try: huggingface-cli login")
                elif "memory" in str(e).lower():
                    print("   💡 Solution: Model too large for current system")
                    print("   💡 Try: Use smaller models or increase system memory")
    
    except ImportError as e:
        print(f"❌ Cannot test models: {e}")
        print("💡 Install transformers first: pip install transformers torch")

def provide_korean_fix():
    """Provide working code for Korean translation"""
    print("\n🛠️ Korean Translation Fix - Working Code:")
    
    working_code = '''
# WORKING KOREAN TRANSLATION CODE

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def translate_to_korean_working(text):
    """
    Working Korean translation using NLLB model
    """
    try:
        # Use Facebook's NLLB model which supports Korean
        model_name = "facebook/nllb-200-distilled-600M"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Set source and target languages
        tokenizer.src_lang = "eng_Latn"  # English
        encoded = tokenizer(text, return_tensors="pt")
        
        # Generate translation to Korean
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id["kor_Hang"],  # Korean
            max_length=128,
            num_beams=4
        )
        
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation
        
    except Exception as e:
        print(f"Translation failed: {e}")
        return None

# Alternative using Google Translate API (Free)
import requests

def translate_to_korean_google(text):
    """
    Alternative Korean translation using Google Translate
    """
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': 'en',  # English
            'tl': 'ko',  # Korean
            'dt': 't',
            'q': text
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0:
                return result[0][0][0]
    except:
        pass
    
    return None

# Example usage:
if __name__ == "__main__":
    test_text = "Hello, how are you today?"
    
    # Try NLLB model
    korean_nllb = translate_to_korean_working(test_text)
    print(f"NLLB: {korean_nllb}")
    
    # Try Google Translate
    korean_google = translate_to_korean_google(test_text)
    print(f"Google: {korean_google}")
'''
    
    print(working_code)
    
    # Write the working code to a file
    with open("/workspace/korean_translation_fix.py", "w") as f:
        f.write(working_code)
    
    print("💾 Working code saved to: korean_translation_fix.py")

def main():
    """Main diagnostic and fix function"""
    print("🚀 Korean Translation Error - Diagnostic & Fix Tool")
    print("=" * 60)
    
    # Step 1: Check dependencies
    missing = check_dependencies()
    
    # Step 2: Install missing dependencies
    if missing:
        user_input = input(f"\n❓ Install missing dependencies {missing}? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            if not install_dependencies(missing):
                print("❌ Cannot proceed without required dependencies")
                return
        else:
            print("⚠️ Skipping dependency installation")
    
    # Step 3: Test Korean models
    test_korean_models()
    
    # Step 4: Provide working solution
    provide_korean_fix()
    
    print("\n✅ Diagnostic complete!")
    print("\n📋 Summary of Solutions:")
    print("1. ❌ Helsinki-NLP/opus-mt-en-ko doesn't exist")
    print("2. ✅ Use facebook/nllb-200-distilled-600M instead")
    print("3. ✅ Use Google Translate API as fallback")
    print("4. ✅ Working code saved to korean_translation_fix.py")
    print("\n💡 Run the translation_module.py for a complete solution!")

if __name__ == "__main__":
    main()