# 📊 Workforce Distribution AI

A comprehensive machine learning application built with Streamlit that predicts employee retention, salary growth, and job role classification using advanced ML algorithms.

## 🚀 Features

- **Employee Retention Prediction**: Predict whether an employee will stay or leave
- **Salary Growth Forecasting**: Estimate next year's salary based on various factors
- **Role Classification**: Automatically classify employees into appropriate job roles
- **Interactive Visualizations**: Real-time charts showing wage growth trends
- **User-Friendly Interface**: Clean, modern Streamlit interface

## 🏗️ Project Structure

```
├── main.py                    # Main Streamlit application
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── Employee.csv              # Dataset
├── model.pkl                 # Trained retention model
├── salary_predictor.pkl      # Trained salary prediction model
├── role_classifier.pkl       # Trained role classification model
├── role_encoder.pkl          # Label encoder for roles
├── imputer.pkl              # Data imputer for missing values
└── README.md                # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd workforce-distribution-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (if needed):**
   ```bash
   python train_model.py
   ```

4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:8501`

## 🌐 Streamlit Cloud Deployment

### Method 1: Direct Deployment
1. Fork this repository to your GitHub account
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and set:
   - **Main file path**: `main.py`
   - **Python version**: 3.9
6. Click "Deploy"

### Method 2: Manual Upload
1. Create a new app on Streamlit Cloud
2. Upload all files from this project
3. Set `main.py` as the main file
4. Deploy

## 🐳 Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t workforce-ai .
docker run -p 8501:8501 workforce-ai
```

## 📊 Model Information

### 1. Employee Retention Model (`model.pkl`)
- **Algorithm**: Random Forest Classifier
- **Purpose**: Predicts if an employee will leave (1) or stay (0)
- **Features**: All input features including salary, experience, age, etc.

### 2. Salary Prediction Model (`salary_predictor.pkl`)
- **Algorithm**: Linear Regression
- **Purpose**: Predicts next year's expected salary
- **Output**: Continuous salary value

### 3. Role Classification Model (`role_classifier.pkl`)
- **Algorithm**: Random Forest Classifier
- **Purpose**: Classifies employees into job roles:
  - Senior Engineer
  - Mid-Level Developer
  - Junior Associate

## 📈 Usage

1. **Input Employee Data:**
   - Joining Year (2000-2025)
   - Payment Tier (1, 2, or 3)
   - Age (18-60)
   - Experience in Current Domain (0-20 years)
   - Current Salary
   - Expected Next Year Salary
   - Education Level

2. **Get Predictions:**
   - Click "Predict Retention & Role"
   - View retention prediction
   - See predicted job role
   - Check salary forecast

3. **Analyze Trends:**
   - View the wage growth visualization
   - Understand experience vs. salary relationships

## 🔧 Configuration

### Environment Variables
For production deployment, consider setting:
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Streamlit Config
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
headless = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 🧪 Model Retraining

To retrain models with new data:

1. Update `Employee.csv` with new data
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. Restart the Streamlit app

## 📊 Data Format

The `Employee.csv` should contain the following columns:
- `Education`: High School, Bachelors, Masters, PHD
- `JoiningYear`: Year employee joined
- `City`: Employee location
- `PaymentTier`: 1, 2, or 3
- `Age`: Employee age
- `Gender`: Male/Female
- `EverBenched`: Yes/No
- `ExperienceInCurrentDomain`: Years of experience
- `LeaveOrNot`: 0 (stay) or 1 (leave)
- `CurrentSalary`: Current salary amount
- `ExpectedNextYearSalary`: Expected salary next year

## 🛠️ Troubleshooting

### Common Issues:

1. **Module Import Errors:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Model Loading Errors:**
   - Ensure all `.pkl` files are in the same directory as `main.py`
   - Run `train_model.py` to regenerate models

3. **Data Loading Errors:**
   - Verify `Employee.csv` exists and has correct format
   - Check file permissions

4. **Streamlit Cloud Deployment Issues:**
   - Ensure all files are pushed to GitHub
   - Check requirements.txt formatting
   - Verify Python version compatibility

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For support and questions, please open an issue in the GitHub repository.

---

Built with ❤️ using Streamlit and scikit-learn