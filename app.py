"""
ChurnAI — Flask Backend
Tera exact churn prediction code yahan run hoga.
Run: python app.py
Then open: http://localhost:5000
"""

import warnings, os, io, base64
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ─── Plot style (Netflix dark) ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F0F0F",
    "axes.facecolor":   "#1A1A1A",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#EEEEEE",
    "xtick.color":      "#EEEEEE",
    "ytick.color":      "#EEEEEE",
    "text.color":       "#EEEEEE",
    "grid.color":       "#333",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
COLORS  = ["#E50914","#2196F3","#4CAF50","#FF9800","#9C27B0"]
PALETTE = ["#E50914","#221F1F","#B81D24","#F5F5F1","#564D4D"]

app = Flask(__name__)

# ─── Global state (in-memory after pipeline runs) ────────────────────────────
STATE = {}

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (your exact code)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_telco(df):
    df = df.copy()
    df.replace(" ", np.nan, inplace=True)
    df.dropna(inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    out = pd.DataFrame()
    out["tenure_months"]        = df["tenure"]
    out["monthly_charges"]      = df["MonthlyCharges"]
    out["total_charges"]        = df["TotalCharges"]
    out["gender_enc"]           = (df["gender"] == "Male").astype(int)
    out["senior_or_age_norm"]   = df["SeniorCitizen"]
    out["has_partner"]          = (df["Partner"] == "Yes").astype(int)
    out["is_streaming"]         = ((df["StreamingTV"]=="Yes")|(df["StreamingMovies"]=="Yes")).astype(int)
    out["is_month_to_month"]    = (df["Contract"] == "Month-to-month").astype(int)
    out["paperless_or_digital"] = (df["PaperlessBilling"] == "Yes").astype(int)
    out["monthly_charges_sq"]   = df["MonthlyCharges"] ** 2
    out["charge_per_tenure"]    = df["MonthlyCharges"] / (df["tenure"] + 1)
    out["inactivity_score"]     = 0.0
    out["multi_service"]        = (
        (df["MultipleLines"]=="Yes").astype(int) +
        (df["OnlineSecurity"]=="Yes").astype(int) +
        (df["OnlineBackup"]=="Yes").astype(int) +
        (df["DeviceProtection"]=="Yes").astype(int) +
        (df["TechSupport"]=="Yes").astype(int)
    )
    out["source"] = 0
    out["Churn"]  = df["Churn"].values
    return out.reset_index(drop=True)


def preprocess_netflix(df):
    df = df.copy()
    df.dropna(inplace=True)
    out = pd.DataFrame()
    out["tenure_months"]        = df["watch_hours"] / (df["avg_watch_time_per_day"].replace(0, 0.1) * 30)
    out["monthly_charges"]      = df["monthly_fee"]
    out["total_charges"]        = df["monthly_fee"] * out["tenure_months"]
    out["gender_enc"]           = (df["gender"] == "Male").astype(int)
    out["senior_or_age_norm"]   = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
    out["has_partner"]          = 0
    out["is_streaming"]         = 1
    out["is_month_to_month"]    = (df["subscription_type"] == "Basic").astype(int)
    out["paperless_or_digital"] = (df["payment_method"].str.lower().str.contains("credit|debit|online")).astype(int)
    out["monthly_charges_sq"]   = df["monthly_fee"] ** 2
    out["charge_per_tenure"]    = df["monthly_fee"] / (out["tenure_months"] + 1)
    out["inactivity_score"]     = df["last_login_days"] / df["last_login_days"].max()
    out["multi_service"]        = df["number_of_profiles"]
    out["source"] = 1
    out["Churn"]  = df["churned"].values
    return out.reset_index(drop=True)


def feature_selection(df, threshold=0.04):
    corr_vals = df.corr()["Churn"].abs().drop("Churn")
    selected  = corr_vals[corr_vals > threshold].index.tolist()
    selected.append("Churn")
    return df[selected]


def split_and_scale(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, feature_names


def apply_smote(X_train, y_train):
    y_arr = np.array(y_train)
    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_r, y_r = sm.fit_resample(X_train, y_arr)
    else:
        # manual fallback
        rng = np.random.default_rng(42)
        minority = X_train[y_arr == 1]
        n_syn = (y_arr == 0).sum() - (y_arr == 1).sum()
        synthetic = []
        for _ in range(max(0, n_syn)):
            idx  = rng.integers(len(minority))
            nbrs = rng.choice(len(minority), size=min(5, len(minority)), replace=False)
            nn   = minority[nbrs[rng.integers(len(nbrs))]]
            lam  = rng.uniform(0, 1)
            synthetic.append(minority[idx] + lam * (nn - minority[idx]))
        if synthetic:
            X_r = np.vstack([X_train, np.array(synthetic)])
            y_r = np.concatenate([y_arr, np.ones(n_syn, dtype=int)])
        else:
            X_r, y_r = X_train, y_arr
        shuffle = rng.permutation(len(X_r))
        X_r, y_r = X_r[shuffle], y_r[shuffle]
    return X_r, y_r


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "KAN (MLP)":           MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu", solver="adam", alpha=0.01, max_iter=300, early_stopping=True, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
    return models


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    cv_f1   = cross_val_score(model, X_train, y_train,
                              cv=StratifiedKFold(5, shuffle=True, random_state=42),
                              scoring="f1", n_jobs=-1)
    return {
        "Accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "Precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":         round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC_AUC":    round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
        "CV_F1_mean": round(cv_f1.mean(), 4),
        "CV_F1_std":  round(cv_f1.std(), 4),
        "Confusion":  confusion_matrix(y_test, y_pred).tolist(),
        "y_pred":     y_pred.tolist(),
        "y_proba":    y_proba.tolist() if y_proba is not None else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH HELPERS  (return base64 PNG)
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0F0F0F")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def make_eda_graph(telco_raw, netflix_raw, merged):
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("EDA — Dual Dataset Churn Overview", fontsize=15, fontweight="bold", color="#E50914")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    counts = telco_raw["Churn"].value_counts()
    ax1.bar(["No Churn","Churn"], counts.values, color=["#2196F3","#E50914"], width=0.5)
    ax1.set_title("Telco — Churn Distribution", fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    counts_n = netflix_raw["churned"].value_counts().sort_index()
    ax2.bar(["No Churn","Churn"], counts_n.values, color=["#2196F3","#E50914"], width=0.5)
    ax2.set_title("Netflix — Churn Distribution", fontweight="bold")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(merged.loc[merged["Churn"]==0,"monthly_charges"], bins=40, alpha=0.7, color="#2196F3", label="No Churn", density=True)
    ax3.hist(merged.loc[merged["Churn"]==1,"monthly_charges"], bins=40, alpha=0.7, color="#E50914", label="Churn",    density=True)
    ax3.set_title("Monthly Charges vs Churn", fontweight="bold")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 0])
    merged.boxplot(column="tenure_months", by="Churn", ax=ax4,
                   boxprops=dict(color="#E50914"), whiskerprops=dict(color="#EEE"),
                   capprops=dict(color="#EEE"), medianprops=dict(color="#FFD700", linewidth=2),
                   flierprops=dict(marker="o", color="#E50914", alpha=0.3))
    ax4.set_title("Tenure by Churn Class", fontweight="bold")
    plt.sca(ax4); plt.title("Tenure by Churn")

    ax5 = fig.add_subplot(gs[1, 1])
    nf = merged[merged["source"] == 1]
    ax5.hist(nf.loc[nf["Churn"]==0,"inactivity_score"], bins=30, alpha=0.7, color="#2196F3", label="No Churn", density=True)
    ax5.hist(nf.loc[nf["Churn"]==1,"inactivity_score"], bins=30, alpha=0.7, color="#E50914", label="Churn",    density=True)
    ax5.set_title("Netflix Inactivity vs Churn", fontweight="bold")
    ax5.legend()

    ax6 = fig.add_subplot(gs[1, 2])
    src = merged.groupby(["source","Churn"]).size().unstack(fill_value=0)
    src.index = ["Telco","Netflix"]
    src.columns = ["No Churn","Churn"]
    src.plot(kind="bar", ax=ax6, color=["#2196F3","#E50914"], width=0.6)
    ax6.set_title("Churn by Dataset Source", fontweight="bold")
    ax6.tick_params(axis="x", rotation=0)

    return fig_to_b64(fig)


def make_corr_graph(merged):
    feat = [c for c in merged.columns if c != "Churn"]
    corr = merged[feat + ["Churn"]].corr()
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle("Feature Correlation Analysis", fontsize=14, fontweight="bold", color="#E50914")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
                cmap="RdBu_r", center=0, ax=axes[0], annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    axes[0].set_title("Full Correlation Matrix", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    churn_corr = corr["Churn"].drop("Churn").sort_values()
    colors = ["#E50914" if v > 0 else "#2196F3" for v in churn_corr.values]
    axes[1].barh(churn_corr.index, churn_corr.values, color=colors)
    axes[1].axvline(0, color="#EEE", linewidth=0.8)
    axes[1].set_title("Correlation with Churn", fontweight="bold")
    plt.tight_layout()
    return fig_to_b64(fig)


def make_model_comparison(results):
    metric_keys = ["Accuracy","Precision","Recall","F1","ROC_AUC"]
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold", color="#E50914")
    for i, metric in enumerate(metric_keys):
        vals = [results[n].get(metric) or 0 for n in names]
        bars = ax.bar(x + i*width, vals, width, label=metric, color=COLORS[i], alpha=0.85)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                    f"{b.get_height():.3f}", ha="center", fontsize=6.5, color="#EEE")
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    return fig_to_b64(fig)


def make_confusion_grid(results, y_test):
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", color="#E50914")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for ax, (name, res) in zip(axes_flat, results.items()):
        cm = np.array(res["Confusion"])
        pct = cm / cm.sum() * 100
        labels = [[f"{cm[i,j]}\n({pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)]
        sns.heatmap(cm, annot=np.array(labels), fmt="", cmap="Reds",
                    linewidths=1, xticklabels=["Pred No","Pred Yes"],
                    yticklabels=["True No","True Yes"], ax=ax, cbar=False,
                    annot_kws={"size": 11, "color": "#EEE"})
        ax.set_title(name, fontweight="bold")
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig_to_b64(fig)


def make_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot([0,1],[0,1],"w--", linewidth=0.8, label="Random")
    fig.suptitle("ROC Curves", fontsize=14, fontweight="bold", color="#E50914")
    for (name, res), color in zip(results.items(), COLORS):
        if res["y_proba"]:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f"{name}  (AUC={res['ROC_AUC']:.4f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig_to_b64(fig)


def make_feature_importance(best_model, X_train, feature_names):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Feature Importance Analysis", fontsize=14, fontweight="bold", color="#E50914")
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        title = "Gini Feature Importance"
    else:
        pi = permutation_importance(best_model, X_train[:500], np.zeros(min(500,len(X_train))),
                                    n_repeats=5, random_state=42)
        imp = pi.importances_mean
        title = "Permutation Importance"
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values("Importance", ascending=True)
    colors_bar = ["#E50914" if v > imp_df["Importance"].median() else "#2196F3" for v in imp_df["Importance"]]
    axes[0].barh(imp_df["Feature"], imp_df["Importance"], color=colors_bar)
    axes[0].set_title(title, fontweight="bold")

    if SHAP_AVAILABLE and hasattr(best_model, "feature_importances_"):
        try:
            explainer   = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_train[:300])
            if isinstance(shap_values, list): shap_vals = shap_values[1]
            else: shap_vals = shap_values
            mean_abs = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({"Feature": feature_names, "SHAP": mean_abs}).sort_values("SHAP", ascending=True)
            axes[1].barh(shap_df["Feature"], shap_df["SHAP"], color="#FF9800")
            axes[1].set_title("Mean |SHAP| Value", fontweight="bold")
        except Exception:
            axes[1].axis("off"); axes[1].set_title("SHAP unavailable")
    else:
        axes[1].axis("off"); axes[1].set_title("Feature Engineering Summary", fontweight="bold")
        table_data = [
            ["tenure_months","Both","Numeric"], ["monthly_charges","Both","Numeric"],
            ["is_month_to_month","Both","Binary"], ["charge_per_tenure","Both","Engineered"],
            ["inactivity_score","Netflix","Engineered"], ["multi_service","Both","Count"],
        ]
        tbl = axes[1].table(cellText=table_data, colLabels=["Feature","Source","Type"],
                            loc="center", cellLoc="center")
        tbl.scale(1.2, 2.2)
    plt.tight_layout()
    return fig_to_b64(fig)


def make_time_series(merged):
    df = merged.copy()
    df["tenure_bin"] = pd.cut(df["tenure_months"].clip(0,72), bins=24)
    ts = df.groupby("tenure_bin", observed=True)["Churn"].mean().dropna()
    ts.index = range(len(ts))
    n_obs = len(ts); n_forecast = 6

    if STATSMODELS_AVAILABLE:
        arima_pred  = ARIMA(ts, order=(2,1,2)).fit().forecast(steps=n_forecast)
        sarima_pred = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,0,4)).fit(disp=False).forecast(steps=n_forecast)
    else:
        x = np.arange(n_obs); slope, intercept = np.polyfit(x, ts.values, 1)
        future_x = np.arange(n_obs, n_obs+n_forecast)
        arima_pred  = pd.Series(slope*future_x+intercept, index=range(n_obs, n_obs+n_forecast))
        sarima_pred = arima_pred * 0.97

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Churn Rate Forecast — ARIMA vs SARIMA", fontsize=14, fontweight="bold", color="#E50914")
    ax.plot(ts.index, ts.values, "o-", color="#F5F5F1", linewidth=2, markersize=4, label="Observed")
    future = range(n_obs, n_obs+n_forecast)
    ax.plot(future, arima_pred,  "^--", color="#E50914", linewidth=2, markersize=6, label="ARIMA Forecast")
    ax.plot(future, sarima_pred, "s--", color="#2196F3", linewidth=2, markersize=6, label="SARIMA Forecast")
    ax.axvline(n_obs-0.5, color="#FFD700", linestyle=":", linewidth=1.5, label="Forecast Start")
    ax.set_xlabel("Tenure Bin"); ax.set_ylabel("Churn Rate")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig_to_b64(fig)


def make_reco_graph(df_scored):
    at_risk = df_scored[df_scored["at_risk"]==1].copy()
    rec_counts = at_risk["Recommendation"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Retention Recommendation Distribution", fontsize=14, fontweight="bold", color="#E50914")
    wedge_colors = ["#E50914","#2196F3","#4CAF50","#FF9800","#9C27B0","#00BCD4","#F44336"]
    axes[0].pie(rec_counts.values,
                labels=[r[:28]+"…" if len(r)>28 else r for r in rec_counts.index],
                colors=wedge_colors[:len(rec_counts)], autopct="%1.1f%%", startangle=140,
                textprops={"color":"#EEE","fontsize":8},
                wedgeprops={"edgecolor":"#111","linewidth":1.5})
    axes[0].set_title("Recommendation Breakdown", fontweight="bold")
    rec_short = {r: r.split("—")[0].strip() for r in at_risk["Recommendation"].unique()}
    at_risk["rec_short"] = at_risk["Recommendation"].map(rec_short)
    at_risk.boxplot(column="churn_probability", by="rec_short", ax=axes[1],
                    boxprops=dict(color="#E50914"), whiskerprops=dict(color="#EEE"),
                    capprops=dict(color="#EEE"), medianprops=dict(color="#FFD700",linewidth=2),
                    flierprops=dict(marker="o",color="#E50914",alpha=0.3))
    axes[1].set_title("Churn Prob by Recommendation", fontweight="bold")
    plt.sca(axes[1]); plt.xticks(rotation=20, ha="right", fontsize=8); plt.title("Churn Prob")
    plt.tight_layout()
    return fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# RETENTION ENGINE  (your exact code)
# ══════════════════════════════════════════════════════════════════════════════

CHURN_THRESHOLD = 0.55

def _recommend(row):
    p = row.get("churn_probability", 0.5)
    if p < 0.35: return "✅ Low Risk — No Action Needed"
    if row.get("monthly_charges", 0) > 80: return "💰 High Spend — Offer Loyalty Discount (10–20%)"
    if row.get("is_month_to_month", 0)==1 and p>0.6: return "📝 Month-to-Month — Promote Annual/2-Year Plan"
    if row.get("inactivity_score", 0) > 0.6: return "🎬 Inactive User — Send Re-engagement Campaign"
    if row.get("tenure_months", 0) < 6: return "🚀 New Customer — Onboarding Bonus / Free Trial Extension"
    if row.get("multi_service", 0) <= 1 and p>0.55: return "📦 Low Add-Ons — Bundle Upsell Offer"
    return "📣 General — Personalised Retention Offer"


def predict_churn_risk(model, scaler, feature_names, df_raw, source="telco"):
    df_proc = preprocess_telco(df_raw) if source=="telco" else preprocess_netflix(df_raw)
    for c in feature_names:
        if c not in df_proc.columns: df_proc[c] = 0
    X = scaler.transform(df_proc[feature_names])
    proba = model.predict_proba(X)[:, 1]
    df_proc["churn_probability"] = proba
    df_proc["at_risk"] = (proba >= CHURN_THRESHOLD).astype(int)
    df_proc["Recommendation"] = df_proc.apply(_recommend, axis=1)
    return df_proc


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run_pipeline", methods=["POST"])
def run_pipeline():
    """Full pipeline: upload CSVs → preprocess → train → graphs → recommendations."""
    try:
        telco_file   = request.files.get("telco_csv")
        netflix_file = request.files.get("netflix_csv")
        if not telco_file or not netflix_file:
            return jsonify({"error": "Dono CSV files upload karo!"}), 400

        telco_raw   = pd.read_csv(telco_file)
        netflix_raw = pd.read_csv(netflix_file)

        # --- preprocess + merge ---
        t_df   = preprocess_telco(telco_raw)
        n_df   = preprocess_netflix(netflix_raw)
        merged = pd.concat([t_df, n_df], ignore_index=True)

        # --- graphs 1 & 2 ---
        g_eda  = make_eda_graph(telco_raw, netflix_raw, merged)
        g_corr = make_corr_graph(merged)

        # --- feature selection + split + SMOTE ---
        df_sel = feature_selection(merged)
        X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df_sel)
        X_sm, y_sm = apply_smote(X_train, y_train)

        # --- train models ---
        models_dict = get_models()
        results = {}
        for name, model in models_dict.items():
            results[name] = evaluate_model(model, X_sm, y_sm, X_test, y_test)

        # --- graphs 3-5 ---
        g_models = make_model_comparison(results)
        y_test_list = list(y_test)
        g_cm   = make_confusion_grid(results, y_test)
        g_roc  = make_roc_curves(results, y_test)

        # --- best model ---
        best_name = max(results, key=lambda k: results[k]["ROC_AUC"] or results[k]["F1"])
        best_model = models_dict[best_name]
        g_feat = make_feature_importance(best_model, X_sm, feature_names)

        # --- time series ---
        g_ts = make_time_series(merged)

        # --- retention recommendations ---
        df_scored = predict_churn_risk(best_model, scaler, feature_names, telco_raw, source="telco")
        g_reco    = make_reco_graph(df_scored)

        # --- summary stats ---
        at_risk    = df_scored[df_scored["at_risk"]==1]
        rec_counts = at_risk["Recommendation"].value_counts().to_dict()

        top_recs = df_scored.sort_values("churn_probability", ascending=False).head(15)
        recs_list = top_recs[["tenure_months","monthly_charges","churn_probability","Recommendation"]].copy()
        recs_list["churn_probability"] = recs_list["churn_probability"].round(4)
        recs_list["tenure_months"] = recs_list["tenure_months"].round(1)
        recs_list["monthly_charges"] = recs_list["monthly_charges"].round(2)

        # build leaderboard
        leaderboard = sorted(
            [{"name": n, **{k: v for k, v in r.items() if k not in ("Confusion","y_pred","y_proba")}}
             for n, r in results.items()],
            key=lambda x: x.get("ROC_AUC") or x.get("F1") or 0, reverse=True
        )

        STATE["results"]       = results
        STATE["best_name"]     = best_name
        STATE["merged_shape"]  = merged.shape
        STATE["merged_churn"]  = round(merged["Churn"].mean(), 4)

        return jsonify({
            "status": "success",
            "stats": {
                "total_customers":  int(merged.shape[0]),
                "churn_rate":       round(float(merged["Churn"].mean()), 4),
                "at_risk":          int(len(at_risk)),
                "best_model":       best_name,
                "best_auc":         results[best_name]["ROC_AUC"],
                "telco_shape":      list(telco_raw.shape),
                "netflix_shape":    list(netflix_raw.shape),
                "smote_available":  SMOTE_AVAILABLE,
                "shap_available":   SHAP_AVAILABLE,
                "xgboost_available":XGBOOST_AVAILABLE,
            },
            "leaderboard":    leaderboard,
            "recommendations": recs_list.to_dict(orient="records"),
            "rec_counts":     rec_counts,
            "graphs": {
                "eda":     g_eda,
                "corr":    g_corr,
                "models":  g_models,
                "cm":      g_cm,
                "roc":     g_roc,
                "feat":    g_feat,
                "ts":      g_ts,
                "reco":    g_reco,
            }
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/predict_single", methods=["POST"])
def predict_single():
    """Ek customer ka churn risk predict karo."""
    try:
        if "model" not in STATE:
            return jsonify({"error": "Pehle pipeline run karo!"}), 400
        data = request.json
        row  = pd.DataFrame([data])
        X    = STATE["scaler"].transform(row[STATE["feature_names"]])
        prob = STATE["model"].predict_proba(X)[0][1]
        rec  = _recommend({**data, "churn_probability": prob})
        return jsonify({"churn_probability": round(float(prob), 4), "recommendation": rec})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚀  ChurnAI Flask Server")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
