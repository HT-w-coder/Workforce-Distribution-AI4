"""
Robust Translation Module with Error Handling
Supports multiple translation backends with fallback options
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Tuple
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationManager:
    """
    A robust translation manager with multiple backends and error handling
    """
    
    # Updated model mappings with correct identifiers
    MARIAN_MODELS = {
        'en-es': 'Helsinki-NLP/opus-mt-en-es',
        'en-fr': 'Helsinki-NLP/opus-mt-en-fr', 
        'en-de': 'Helsinki-NLP/opus-mt-en-de',
        'en-it': 'Helsinki-NLP/opus-mt-en-it',
        'en-pt': 'Helsinki-NLP/opus-mt-en-roa',  # Romance languages including Portuguese
        'en-ru': 'Helsinki-NLP/opus-mt-en-ru',
        'en-zh': 'Helsinki-NLP/opus-mt-en-zh',
        'en-ja': 'Helsinki-NLP/opus-mt-en-jap',
        'en-ar': 'Helsinki-NLP/opus-mt-en-ar',
        'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
        'es-en': 'Helsinki-NLP/opus-mt-es-en',
        'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
        'de-en': 'Helsinki-NLP/opus-mt-de-en',
        'ru-en': 'Helsinki-NLP/opus-mt-ru-en',
        'zh-en': 'Helsinki-NLP/opus-mt-zh-en',
    }
    
    # Language codes and names
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'ko': 'Korean',
        'tr': 'Turkish',
        'nl': 'Dutch'
    }
    
    def __init__(self):
        """Initialize the translation manager"""
        self.transformers_available = False
        self.loaded_models = {}
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize available translation backends"""
        try:
            # Try to import transformers
            from transformers import MarianMTModel, MarianTokenizer, AutoModel, AutoTokenizer
            self.transformers_available = True
            self.MarianMTModel = MarianMTModel
            self.MarianTokenizer = MarianTokenizer  
            self.AutoModel = AutoModel
            self.AutoTokenizer = AutoTokenizer
            logger.info("✅ Transformers library loaded successfully")
        except ImportError:
            logger.warning("⚠️ Transformers library not available")
            self.transformers_available = False
    
    def _load_marian_model(self, model_name: str) -> Tuple[Optional[object], Optional[object]]:
        """
        Load MarianMT model with proper error handling
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Tuple of (model, tokenizer) or (None, None) if failed
        """
        if not self.transformers_available:
            return None, None
            
        try:
            logger.info(f"Loading model: {model_name}")
            
            # First try MarianMT
            try:
                tokenizer = self.MarianTokenizer.from_pretrained(model_name)
                model = self.MarianMTModel.from_pretrained(model_name)
                logger.info(f"✅ Successfully loaded MarianMT model: {model_name}")
                return model, tokenizer
            except Exception as e:
                logger.warning(f"MarianMT failed for {model_name}: {str(e)}")
                
                # Fallback to AutoModel
                try:
                    tokenizer = self.AutoTokenizer.from_pretrained(model_name)
                    model = self.AutoModel.from_pretrained(model_name)
                    logger.info(f"✅ Successfully loaded AutoModel: {model_name}")
                    return model, tokenizer
                except Exception as e2:
                    logger.error(f"AutoModel also failed for {model_name}: {str(e2)}")
                    return None, None
                    
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return None, None
    
    def _get_model_key(self, source_lang: str, target_lang: str) -> Optional[str]:
        """Get the model key for language pair"""
        key = f"{source_lang}-{target_lang}"
        if key in self.MARIAN_MODELS:
            return key
        
        # Try reverse direction if available
        reverse_key = f"{target_lang}-{source_lang}"
        if reverse_key in self.MARIAN_MODELS:
            return reverse_key
            
        return None
    
    def translate_with_transformers(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text using transformers library
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or None if failed
        """
        if not self.transformers_available:
            return None
            
        model_key = self._get_model_key(source_lang, target_lang)
        if not model_key:
            logger.warning(f"No model available for {source_lang} -> {target_lang}")
            return None
            
        model_name = self.MARIAN_MODELS[model_key]
        
        # Load model if not already loaded
        if model_name not in self.loaded_models:
            model, tokenizer = self._load_marian_model(model_name)
            if model is None or tokenizer is None:
                logger.error(f"Failed to load model for {source_lang} -> {target_lang}")
                return None
            self.loaded_models[model_name] = (model, tokenizer)
        
        try:
            model, tokenizer = self.loaded_models[model_name]
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Generate translation
            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            
            # Decode output
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"✅ Translation successful: {source_lang} -> {target_lang}")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return None
    
    def translate_with_google_api(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate using Google Translate API (free tier)
        Note: This is a basic implementation - for production use proper API key
        """
        try:
            # This is a simplified approach - in production, use official Google Translate API
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                'client': 'gtx',
                'sl': source_lang,
                'tl': target_lang,
                'dt': 't',
                'q': text
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0 and len(result[0]) > 0:
                    translated = result[0][0][0]
                    logger.info(f"✅ Google Translate successful: {source_lang} -> {target_lang}")
                    return translated
                    
        except Exception as e:
            logger.error(f"Google Translate failed: {str(e)}")
            
        return None
    
    def translate(self, text: str, source_lang: str, target_lang: str, method: str = "auto") -> Dict[str, str]:
        """
        Translate text with fallback options
        
        Args:
            text: Text to translate
            source_lang: Source language code  
            target_lang: Target language code
            method: Translation method ("transformers", "google", "auto")
            
        Returns:
            Dictionary with translation result and metadata
        """
        if not text.strip():
            return {"error": "Empty text provided"}
            
        if source_lang == target_lang:
            return {"translation": text, "method": "no_translation_needed"}
        
        result = {"original": text, "source_lang": source_lang, "target_lang": target_lang}
        
        # Try different methods based on preference
        if method in ["transformers", "auto"]:
            translation = self.translate_with_transformers(text, source_lang, target_lang)
            if translation:
                result["translation"] = translation
                result["method"] = "transformers"
                return result
        
        if method in ["google", "auto"]:
            translation = self.translate_with_google_api(text, source_lang, target_lang)
            if translation:
                result["translation"] = translation  
                result["method"] = "google_translate"
                return result
        
        result["error"] = f"Translation failed for {source_lang} -> {target_lang}"
        return result
    
    def get_available_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.SUPPORTED_LANGUAGES
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available translation pairs"""
        pairs = []
        for key in self.MARIAN_MODELS.keys():
            source, target = key.split('-')
            if source in self.SUPPORTED_LANGUAGES and target in self.SUPPORTED_LANGUAGES:
                pairs.append(f"{self.SUPPORTED_LANGUAGES[source]} → {self.SUPPORTED_LANGUAGES[target]}")
        return sorted(pairs)


def create_translation_interface():
    """
    Create Streamlit interface for translation functionality
    """
    st.header("🌍 Multi-Language Translation")
    
    # Initialize translation manager
    if 'translation_manager' not in st.session_state:
        st.session_state.translation_manager = TranslationManager()
    
    translator = st.session_state.translation_manager
    languages = translator.get_available_languages()
    
    # Create layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source")
        source_lang = st.selectbox(
            "Source Language:",
            options=list(languages.keys()),
            format_func=lambda x: f"{languages[x]} ({x})",
            key="source_lang"
        )
        
        source_text = st.text_area(
            "Enter text to translate:",
            placeholder="Type your text here...",
            height=150,
            key="source_text"
        )
    
    with col2:
        st.subheader("Translation")
        target_lang = st.selectbox(
            "Target Language:",
            options=list(languages.keys()),
            format_func=lambda x: f"{languages[x]} ({x})",
            key="target_lang",
            index=1  # Default to second language
        )
        
        # Translation method selection
        method = st.selectbox(
            "Translation Method:",
            options=["auto", "transformers", "google"],
            format_func=lambda x: {
                "auto": "Auto (Try transformers, fallback to Google)",
                "transformers": "Transformers (MarianMT models)",
                "google": "Google Translate"
            }[x]
        )
    
    # Translate button
    if st.button("🔄 Translate", type="primary"):
        if source_text.strip():
            with st.spinner("Translating..."):
                result = translator.translate(
                    text=source_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    method=method
                )
                
                if "translation" in result:
                    st.success("✅ Translation completed!")
                    st.text_area(
                        "Translation:",
                        value=result["translation"],
                        height=150,
                        key="translation_result"
                    )
                    
                    # Show metadata
                    st.info(f"Method used: {result.get('method', 'unknown')}")
                    
                elif "error" in result:
                    st.error(f"❌ {result['error']}")
                    
                    # Show available pairs if model not found
                    if "No model available" in result['error']:
                        st.write("Available translation pairs:")
                        pairs = translator.get_available_pairs()
                        for pair in pairs[:10]:  # Show first 10
                            st.write(f"• {pair}")
        else:
            st.warning("Please enter text to translate")
    
    # Show system status
    with st.expander("🔧 System Status"):
        st.write(f"**Transformers Available:** {'✅ Yes' if translator.transformers_available else '❌ No'}")
        st.write(f"**Loaded Models:** {len(translator.loaded_models)}")
        
        if translator.loaded_models:
            st.write("**Currently Loaded Models:**")
            for model_name in translator.loaded_models.keys():
                st.write(f"• {model_name}")


def fix_korean_translation_error():
    """
    Provide specific fix for the Korean translation error
    """
    st.header("🔧 Korean Translation Fix")
    
    st.error("**Error Found:** `Helsinki-NLP/opus-mt-en-ko` model not available")
    
    st.write("### 🛠️ Solutions:")
    
    st.write("**1. Use Alternative Korean Model:**")
    st.code("""
# Try these alternative models for Korean:
- Helsinki-NLP/opus-mt-en-ko  # May not exist
- Facebook/nllb-200-distilled-600M  # Multilingual model with Korean
- google/mt5-small  # Google's multilingual model
""")
    
    st.write("**2. Install Required Dependencies:**")
    st.code("""
pip install transformers torch sentencepiece
# or
pip install transformers[torch]
""")
    
    st.write("**3. Alternative Code Implementation:**")
    st.code("""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Alternative approach for Korean translation
def translate_to_korean(text):
    try:
        # Try Facebook's NLLB model (supports Korean)
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Tokenize with language codes
        inputs = tokenizer(text, return_tensors="pt")
        
        # Generate translation (English to Korean)
        outputs = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.lang_code_to_id["kor_Hang"],
            max_length=128
        )
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
        
    except Exception as e:
        print(f"Translation failed: {e}")
        return None
""")
    
    st.write("**4. Quick Test:**")
    if st.button("Test Korean Translation Fix"):
        st.info("Testing alternative Korean translation approach...")
        
        # Test with our translation manager
        if 'translation_manager' not in st.session_state:
            st.session_state.translation_manager = TranslationManager()
        
        translator = st.session_state.translation_manager
        
        # Try translating to Korean using fallback
        result = translator.translate(
            text="Hello, how are you?",
            source_lang="en", 
            target_lang="ko",
            method="google"  # Use Google as fallback
        )
        
        if "translation" in result:
            st.success(f"✅ Test successful! Translation: {result['translation']}")
        else:
            st.error("❌ Test failed. Korean translation not available.")


if __name__ == "__main__":
    # For testing the module directly
    st.set_page_config(page_title="Translation Module", layout="wide")
    
    tab1, tab2 = st.tabs(["Translation", "Korean Fix"])
    
    with tab1:
        create_translation_interface()
    
    with tab2:
        fix_korean_translation_error()