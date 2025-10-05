"""
Simple Korean Translation Fix - No Heavy Dependencies
Uses Google Translate API and lighter alternatives
"""

import streamlit as st
import requests
import json
from typing import Optional, Dict
import time

class SimpleKoreanTranslator:
    """
    Simple Korean translator using Google Translate API
    No heavy ML dependencies required
    """
    
    def __init__(self):
        self.base_url = "https://translate.googleapis.com/translate_a/single"
        
    def translate_to_korean(self, text: str) -> Optional[str]:
        """
        Translate English text to Korean using Google Translate
        
        Args:
            text: English text to translate
            
        Returns:
            Korean translation or None if failed
        """
        try:
            params = {
                'client': 'gtx',
                'sl': 'en',  # Source: English
                'tl': 'ko',  # Target: Korean  
                'dt': 't',   # Return translation
                'q': text
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0 and len(result[0]) > 0:
                    return result[0][0][0]
            
            return None
            
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return None
    
    def translate_from_korean(self, text: str) -> Optional[str]:
        """
        Translate Korean text to English
        
        Args:
            text: Korean text to translate
            
        Returns:
            English translation or None if failed
        """
        try:
            params = {
                'client': 'gtx',
                'sl': 'ko',  # Source: Korean
                'tl': 'en',  # Target: English
                'dt': 't',
                'q': text
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0 and len(result[0]) > 0:
                    return result[0][0][0]
            
            return None
            
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return None

def fix_marianmt_error():
    """
    Provide solutions for the MarianMT Korean model error
    """
    st.header("🔧 MarianMT Korean Translation Error Fix")
    
    st.error("**Error:** `Helsinki-NLP/opus-mt-en-ko` model not found")
    
    st.write("### 🎯 Root Cause:")
    st.write("- The model `Helsinki-NLP/opus-mt-en-ko` doesn't exist on HuggingFace Hub")
    st.write("- Korean language code should be `kor_Hang` not `ko` for some models")
    st.write("- MarianMT has limited Korean support")
    
    st.write("### 💡 Solutions:")
    
    # Solution 1: Use Google Translate
    with st.expander("Solution 1: Use Google Translate (Recommended)"):
        st.code("""
# Simple Google Translate approach
import requests

def translate_to_korean(text):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        'client': 'gtx',
        'sl': 'en',
        'tl': 'ko', 
        'dt': 't',
        'q': text
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        result = response.json()
        return result[0][0][0]
    return None
        """)
    
    # Solution 2: Alternative models
    with st.expander("Solution 2: Use Alternative Models"):
        st.write("**Working Models for Korean:**")
        st.code("""
# Use NLLB model (multilingual)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# For Korean: use 'kor_Hang' as language code
tokenizer.src_lang = "eng_Latn"
inputs = tokenizer(text, return_tensors="pt")
generated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["kor_Hang"],
    max_length=128
)
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        """)
    
    # Solution 3: Fix the original code
    with st.expander("Solution 3: Fix Your Existing Code"):
        st.write("**Replace the problematic model with working alternatives:**")
        st.code("""
# BEFORE (doesn't work):
model_name = "Helsinki-NLP/opus-mt-en-ko"

# AFTER (works):
model_name = "facebook/nllb-200-distilled-600M"
# OR
model_name = "google/mt5-small"

# Also update the language codes:
# Korean: 'kor_Hang' (not 'ko')
# English: 'eng_Latn' (not 'en')
        """)

def test_korean_translation():
    """
    Interactive Korean translation tester
    """
    st.header("🧪 Korean Translation Tester")
    
    # Initialize translator
    if 'korean_translator' not in st.session_state:
        st.session_state.korean_translator = SimpleKoreanTranslator()
    
    translator = st.session_state.korean_translator
    
    # Translation interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🇺🇸 English → 🇰🇷 Korean")
        english_text = st.text_area(
            "Enter English text:",
            placeholder="Hello, how are you?",
            height=100
        )
        
        if st.button("Translate to Korean", type="primary"):
            if english_text.strip():
                with st.spinner("Translating..."):
                    korean_result = translator.translate_to_korean(english_text)
                
                if korean_result:
                    st.success("✅ Translation successful!")
                    st.text_area("Korean translation:", value=korean_result, height=100)
                else:
                    st.error("❌ Translation failed")
            else:
                st.warning("Please enter text to translate")
    
    with col2:
        st.subheader("🇰🇷 Korean → 🇺🇸 English")
        korean_text = st.text_area(
            "Enter Korean text:",
            placeholder="안녕하세요, 어떻게 지내세요?",
            height=100
        )
        
        if st.button("Translate to English", type="primary"):
            if korean_text.strip():
                with st.spinner("Translating..."):
                    english_result = translator.translate_from_korean(korean_text)
                
                if english_result:
                    st.success("✅ Translation successful!")
                    st.text_area("English translation:", value=english_result, height=100)
                else:
                    st.error("❌ Translation failed")
            else:
                st.warning("Please enter text to translate")
    
    # Test with sample phrases
    st.subheader("🔍 Quick Tests")
    
    test_phrases = [
        "Hello, world!",
        "How are you today?", 
        "Thank you very much",
        "Good morning",
        "Nice to meet you"
    ]
    
    if st.button("🚀 Run Quick Tests"):
        st.write("**Testing Korean Translation:**")
        
        for phrase in test_phrases:
            with st.spinner(f"Testing: {phrase}"):
                korean = translator.translate_to_korean(phrase)
                if korean:
                    st.write(f"✅ `{phrase}` → `{korean}`")
                else:
                    st.write(f"❌ `{phrase}` → Failed")
                time.sleep(0.5)  # Avoid rate limiting

def create_fixed_requirements():
    """
    Show the corrected requirements.txt without heavy dependencies
    """
    st.header("📄 Fixed Requirements.txt")
    
    st.write("**Lightweight version (no heavy ML dependencies):**")
    st.code("""
streamlit==1.7.0
pandas==2.1.4
scikit-learn==1.3.2
joblib==1.3.2
matplotlib==3.8.2
numpy==1.25.2
requests==2.31.0
    """)
    
    st.write("**Full version (with ML translation models):**")
    st.code("""
streamlit==1.7.0
pandas==2.1.4
scikit-learn==1.3.2
joblib==1.3.2
matplotlib==3.8.2
numpy==1.25.2
transformers==4.35.0
torch>=1.9.0
sentencepiece==0.1.99
requests==2.31.0
    """)
    
    st.info("💡 **Recommendation:** Start with the lightweight version and add ML dependencies only if needed")

def main():
    """
    Main Streamlit app for Korean translation fix
    """
    st.set_page_config(
        page_title="Korean Translation Fix", 
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Korean Translation Error - Complete Fix")
    st.write("Solving the `Helsinki-NLP/opus-mt-en-ko` model error")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Error Fix", 
        "🧪 Test Translation", 
        "📄 Requirements", 
        "💻 Code Examples"
    ])
    
    with tab1:
        fix_marianmt_error()
    
    with tab2:
        test_korean_translation()
    
    with tab3:
        create_fixed_requirements()
    
    with tab4:
        st.header("💻 Complete Code Examples")
        
        st.subheader("Simple Korean Translator Class")
        st.code('''
class KoreanTranslator:
    def __init__(self):
        self.base_url = "https://translate.googleapis.com/translate_a/single"
    
    def translate(self, text, source_lang='en', target_lang='ko'):
        try:
            params = {
                'client': 'gtx',
                'sl': source_lang,
                'tl': target_lang,
                'dt': 't',
                'q': text
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result[0][0][0]
            return None
            
        except Exception as e:
            print(f"Translation error: {e}")
            return None

# Usage:
translator = KoreanTranslator()
korean_text = translator.translate("Hello, world!", 'en', 'ko')
print(korean_text)  # Should print Korean translation
        ''')

if __name__ == "__main__":
    main()