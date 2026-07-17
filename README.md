# 🧠 ChurnAI — Full Stack Churn Prediction Dashboard


---

## 📁 Project Structure

```
churnai/
├── app.py              ← Flask backend 
├── requirements.txt    ← Saari dependencies
├── README.md
└── templates/
    └── index.html      ← Frontend dashboard 
```

---

## 🚀 Setup & Run (Step by Step)

### Step 1 — Dependencies install karo
```bash
cd churnai
pip install -r requirements.txt
```

### Step 2 — Flask server start karo
```bash
python app.py
```

### Step 3 — Browser mein open karo
```
http://localhost:5000
```

### Step 4 — CSV files upload karo
- **Telco CSV** → `Telco-Customer-Churn.csv`
- **Netflix CSV** → `netflix_customer_churn.csv`

### Step 5 — "Run Pipeline" button dabao 🚀

---

## ✅ Full Pipeline

| Step | Work flow |
|------|---------|
| 1 | CSV files load + preview |
| 2 | Telco + Netflix preprocessing  |
| 3 | Merged dataset + feature engineering |
| 4 | EDA graphs (Graph 1 & 2) |
| 5 | Feature selection (correlation threshold) |
| 6 | Train/Test split + StandardScaler |
| 7 | SMOTE class balancing |
| 8 | 5 models train: LR, DT, RF, KAN(MLP), XGBoost |
| 9 | Metrics: Accuracy, Precision, Recall, F1, AUC, CV-F1 |
| 10 | Model comparison graphs (Graph 3, 4, 5) |
| 11 | SHAP / Permutation feature importance (Graph 6) |
| 12 | ARIMA + SARIMA time series forecast (Graph 7) |
| 13 | Retention recommendations engine (Graph 8) |

---

## 📊 Dashboard Sections

- **Dashboard** — Metrics + live log + mini leaderboard
- **All 8 Graphs** — switch between all graphs
- **Recommendations** — At-risk customers with retention strategy
- **Leaderboard** — Full model metrics table

---

## ⚠️ Optional Libraries

Install these for best results:
- `xgboost` → XGBoost model 
- `imbalanced-learn` → SMOTE 
- `shap` → SHAP values 
- `statsmodels` → ARIMA/SARIMA 

---

## 🔧 Troubleshooting

**Port already in use?**
```bash
python app.py  # changes port automatically
# ya manually: app.run(port=5001)
```

**CSV columns mismatch?**
- Telco CSV must have: `tenure`, `MonthlyCharges`, `TotalCharges`, `Churn`, etc.
- Netflix CSV must have: `watch_hours`, `monthly_fee`, `churned`, `age`, etc.
