# 📊 ChurnAI
### AI-Powered Customer Churn Prediction & Retention Analytics Platform

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

</p>

---

# 📌 Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing existing customers directly impacts revenue, making customer retention as important as customer acquisition.

**ChurnAI** is a full-stack Machine Learning platform that predicts customer churn, compares multiple ML models, explains model decisions using SHAP values, forecasts future churn trends, and recommends personalized retention strategies through an interactive Flask dashboard.

The system combines data engineering, machine learning, explainable AI, and business intelligence to help organizations identify high-risk customers before they leave.

---

# ✨ Features

- 📂 Multi-dataset support
- 🧹 Automated preprocessing pipeline
- ⚙ Feature Engineering
- 🎯 Feature Selection
- ⚖ Class imbalance handling using SMOTE
- 🤖 Five Machine Learning models
- 📈 Interactive Dashboard
- 📊 Model Leaderboard
- 🔍 SHAP Explainability
- 📉 Time-Series Forecasting
- 💡 AI-powered Retention Recommendations

---

# 🏗 System Architecture

```

CSV Datasets
│
▼
Data Cleaning
│
▼
Feature Engineering
│
▼
Feature Selection
│
▼
Train/Test Split
│
▼
StandardScaler
│
▼
SMOTE Balancing
│
▼
Machine Learning Models
│
▼
Model Evaluation
│
▼
SHAP Explainability
│
▼
Retention Recommendation Engine
│
▼
Interactive Dashboard

```

---

# 🤖 Machine Learning Pipeline

### Data Sources

- Telco Customer Churn Dataset
- Netflix Customer Churn Dataset

---

### Data Processing

- Missing Value Handling
- Duplicate Removal
- Encoding
- Feature Engineering
- Feature Selection
- Standard Scaling
- SMOTE Oversampling

---

### Models Evaluated

| Model | Purpose |
|---------|-----------|
| Logistic Regression | Baseline |
| Decision Tree | Classification |
| Random Forest | Ensemble |
| Multi-Layer Perceptron (KAN/MLP) | Neural Network |
| XGBoost | Gradient Boosting |

---

# 📊 Dashboard Modules

### Dashboard

- Dataset Preview
- Training Progress
- Live Logs
- Model Leaderboard

### Analytics

- Exploratory Data Analysis
- Customer Distribution
- Correlation Analysis
- Feature Importance

### Explainability

- SHAP Values
- Permutation Importance

### Forecasting

- ARIMA Forecast
- SARIMA Forecast

### Business Intelligence

- Customer Segmentation
- Retention Recommendation Engine

---

# 📈 Model Evaluation

Models are compared using

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross Validation
- Confusion Matrix

The best-performing model is automatically selected and displayed on the leaderboard.

---

# 💡 Retention Recommendation Engine

Beyond predicting customer churn, ChurnAI generates personalized retention strategies based on customer behavior.

Examples include:

- Discount Recommendations
- Loyalty Rewards
- Annual Subscription Offers
- Personalized Engagement Campaigns
- Premium Plan Suggestions

---

# 🛠 Tech Stack

### Backend

- Python
- Flask

### Machine Learning

- Scikit-learn
- XGBoost
- SHAP
- Pandas
- NumPy

### Time-Series

- ARIMA
- SARIMA

### Visualization

- Matplotlib
- Plotly

---

# 📂 Project Structure

```

ChurnAI/
│
├── app.py
├── requirements.txt
├── templates/
├── static/
├── models/
├── assets/
├── README.md
└── Project_Report.pdf

```

---

# 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ChurnAI.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open your browser

```
http://localhost:5000
```

---

# 📸 Screenshots

Add screenshots after uploading them.

- Dashboard
- Leaderboard
- SHAP Graph
- Model Comparison
- Retention Recommendation
- Forecast Graph

---

# 📄 Project Report

A complete project report including implementation details, methodology, evaluation metrics, and future scope is available in this repository.

📘 **Project_Report.pdf**

---

# 🔮 Future Improvements

- Deep Learning Models
- AutoML Integration
- Customer Lifetime Value Prediction
- Real-Time Prediction API
- Docker Support
- Cloud Deployment
- LLM-powered Customer Insights
- Multi-tenant Dashboard

---

# 👩‍💻 Author

**Himanshi Tanwar**

Bachelor of Computer Applications (AI & ML)

Interested in

- Machine Learning
- Explainable AI
- Predictive Analytics
- Business Intelligence
- Artificial Intelligence

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
