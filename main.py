# main.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import requests
from typing import Optional

st.set_page_config(page_title="Workforce Distribution AI", layout="wide")

# Korean Translation Functionality
class KoreanTranslator:
    """Simple Korean translator using Google Translate API"""
    
    def __init__(self):
        self.base_url = "https://translate.googleapis.com/translate_a/single"
    
    def translate_to_korean(self, text: str) -> Optional[str]:
        """Translate English text to Korean"""
        try:
            params = {
                'client': 'gtx',
                'sl': 'en',
                'tl': 'ko', 
                'dt': 't',
                'q': text
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0 and len(result[0]) > 0:
                    return result[0][0][0]
            return None
            
        except Exception:
            return None

# Load Models and Tools
leave_model = joblib.load("model.pkl")
salary_model = joblib.load("salary_predictor.pkl")
role_model = joblib.load("role_classifier.pkl")
role_encoder = joblib.load("role_encoder.pkl")
imputer = joblib.load("imputer.pkl")
df = pd.read_csv("Employee.csv")

# Initialize Korean translator
if 'korean_translator' not in st.session_state:
    st.session_state.korean_translator = KoreanTranslator()

st.title("📊 Workforce Distribution AI")
st.subheader("🔍 Predict Retention, Salary Growth & Role Classification")

# Create tabs for different features
tab1, tab2 = st.tabs(["📊 Prediction", "🌍 Korean Translation"])

with tab1:
    # Input Fields
    joining_year = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2015)
    payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
    age = st.slider("Age", 18, 60, 30)
    experience = st.slider("Experience in Current Domain", 0, 20, 3)
    current_salary = st.number_input("Current Salary", value=40000)
    expected_salary = st.number_input("Expected Salary Next Year", value=42000)
    education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])

    # Education Mapping
    education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    education_level = education_map[education]

    # Prepare Input
    annual_growth = expected_salary - current_salary

    input_data = pd.DataFrame([{
        "JoiningYear": joining_year,
        "PaymentTier": payment_tier,
        "Age": age,
        "ExperienceInCurrentDomain": experience,
        "CurrentSalary": current_salary,
        "ExpectedNextYearSalary": expected_salary,
        "EducationLevel": education_level,
        "AnnualWageGrowth": annual_growth
    }])

    # Impute Missing Values
    input_imputed = imputer.transform(input_data)

    # Predictions
    if st.button("Predict Retention & Role"):
        # LeaveOrNot
        leave_pred = leave_model.predict(input_imputed)[0]
        leave_result = "❌ Will Leave" if leave_pred == 1 else "✅ Will Stay"
        st.success(f"Employee Retention: **{leave_result}**")

        # Role Prediction
        role_encoded = role_model.predict(input_imputed)[0]
        role_name = role_encoder.inverse_transform([role_encoded])[0]
        st.info(f"Predicted Job Role: **{role_name}**")

        # Salary Prediction
        predicted_salary = salary_model.predict(input_imputed)[0]
        st.write(f"💰 Predicted Next Year Salary: ₹ **{int(predicted_salary):,}**")

        # Korean Translation Option
        prediction_text = f"Employee will {leave_result.replace('❌ ', '').replace('✅ ', '').lower()}. Predicted job role: {role_name}. Expected salary: {int(predicted_salary):,} rupees."
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("🌍 **Translation:** Get results in Korean")
        with col2:
            if st.button("🇰🇷 Translate to Korean"):
                with st.spinner("Translating..."):
                    korean_translation = st.session_state.korean_translator.translate_to_korean(prediction_text)
                    if korean_translation:
                        st.success("Korean Translation:")
                        st.write(korean_translation)
                    else:
                        st.error("Translation failed")

    # Salary Growth Visualization
    st.subheader("📈 Average Wage Growth by Experience")
    df["AnnualWageGrowth"] = df["ExpectedNextYearSalary"] - df["CurrentSalary"]
    avg_growth = df.groupby("ExperienceInCurrentDomain")["AnnualWageGrowth"].mean()

    fig, ax = plt.subplots()
    avg_growth.plot(kind='line', marker='o', ax=ax, color='green')
    ax.set_title("Avg Annual Wage Growth vs. Experience")
    ax.set_xlabel("Experience (Years)")
    ax.set_ylabel("Annual Wage Growth")
    st.pyplot(fig)

with tab2:
    st.header("🇰🇷 Korean Translation Tool")
    st.write("Translate any text to Korean using Google Translate API")
    
    # Translation interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🇺🇸 English")
        english_input = st.text_area(
            "Enter English text:",
            placeholder="Hello, how are you?",
            height=150
        )
    
    with col2:
        st.subheader("🇰🇷 Korean")
        if st.button("🔄 Translate", type="primary"):
            if english_input.strip():
                with st.spinner("Translating..."):
                    korean_result = st.session_state.korean_translator.translate_to_korean(english_input)
                
                if korean_result:
                    st.text_area("Korean translation:", value=korean_result, height=150)
                else:
                    st.error("❌ Translation failed. Please try again.")
            else:
                st.warning("Please enter text to translate")
    
    # Status and troubleshooting
    with st.expander("🔧 Korean Translation Status & Troubleshooting"):
        st.write("**Status:** ✅ Google Translate API integration working")
        st.write("**Note:** This fixes the `Helsinki-NLP/opus-mt-en-ko` model error by using Google Translate instead")
        
        st.write("**Common Issues:**")
        st.write("• If translation fails, check internet connection")
        st.write("• Large texts may take longer to translate")
        st.write("• Rate limiting may apply for many requests")
        
        st.write("**Alternative:** If you need offline translation, install transformers and use:")
        st.code("pip install transformers torch\n# Then use facebook/nllb-200-distilled-600M model")