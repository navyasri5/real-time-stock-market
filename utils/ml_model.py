"""
ml_model.py
Trains a Random Forest model to predict next-day price direction.
Explainability via SHAP and LIME.
"""

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

FEATURE_COLS = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_Mid",
    "EMA_20", "EMA_50",
    "Daily_Return", "5D_Return", "20D_Return",
    "Volatility", "Volume"
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Build feature matrix X and target y.
    Target: 1 if next-day close > today's close, else 0.
    """
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df["Target"]
    return X, y, available


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> dict:
    """Train GradientBoosting classifier. Returns model artifacts."""
    X, y, features = prepare_features(df)
    if len(X) < 60:
        return {"error": "Not enough data (need at least 60 rows after feature prep)."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "accuracy": acc,
        "report": report,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_s": X_train_s,
        "X_test_s": X_test_s,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict_next_day(artifacts: dict, df: pd.DataFrame) -> dict:
    """Predict next-day direction for the most recent data point."""
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    features = artifacts["features"]

    latest = df[features].iloc[-1:].values
    latest_s = scaler.transform(latest)

    pred = model.predict(latest_s)[0]
    prob = model.predict_proba(latest_s)[0]

    direction = "UP" if pred == 1 else "DOWN"
    confidence = round(max(prob) * 100, 1)

    return {
        "direction": direction,
        "confidence": confidence,
        "prob_up": round(prob[1] * 100, 1),
        "prob_down": round(prob[0] * 100, 1),
    }


# ─── SHAP EXPLAINABILITY ─────────────────────────────────────────────────────

def get_shap_explanation(artifacts: dict, n_samples: int = 100) -> dict:
    """
    Compute SHAP values using TreeExplainer (fast, native for GradientBoosting).
    Falls back to feature importances if SHAP fails.
    Returns fig (matplotlib) and feature importance series.
    """
    model = artifacts["model"]
    X_train_s = artifacts["X_train_s"]
    X_test_s = artifacts["X_test_s"]
    features = artifacts["features"]

    sample = X_test_s[:min(n_samples, len(X_test_s))]

    try:
        # TreeExplainer works natively with GradientBoostingClassifier
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        # Normalise to 2D: (n_samples, n_features)
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        else:
            sv = np.array(shap_values)

        n_feat = len(features)
        n_samp = sample.shape[0]

        if sv.ndim == 3:
            sv = sv[:, :, 1]                          # (n_samples, n_features, n_classes)
        elif sv.ndim == 2:
            if sv.shape == (n_feat, 2):
                sv = sv[:, 1].reshape(1, n_feat)      # (n_features, 2) squeezed
            elif sv.shape[1] == 2 and sv.shape[0] == n_samp:
                sv = sv[:, 1].reshape(-1, 1)          # (n_samples, 2) — rare
            # else already (n_samples, n_features)

        # sv is now (n_samples, n_features)
        importance = pd.Series(
            np.abs(sv).mean(axis=0), index=features[:sv.shape[1]]
        ).sort_values(ascending=False)

    except Exception:
        # Fallback: use built-in feature importances from the model
        imp = model.feature_importances_
        importance = pd.Series(imp, index=features).sort_values(ascending=False)
        sv = None

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C9A84C"] * len(importance)
    ax.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP Value| (Feature Impact)", fontsize=11)
    ax.set_title("SHAP Feature Importance - What Drives the Prediction?", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(colors="white")
    plt.tight_layout()

    return {"fig": fig, "importance": importance, "shap_values": sv, "sample": sample}


# ─── LIME EXPLAINABILITY ─────────────────────────────────────────────────────

def get_lime_explanation(artifacts: dict, df: pd.DataFrame) -> dict:
    """
    Generate LIME explanation for the most recent prediction.
    Returns explanation object and a matplotlib figure.
    """
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    features = artifacts["features"]
    X_train_s = artifacts["X_train_s"]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_s,
        feature_names=features,
        class_names=["DOWN", "UP"],
        mode="classification",
        random_state=42
    )

    latest = df[features].iloc[-1:].values
    latest_s = scaler.transform(latest)[0]

    exp = explainer.explain_instance(
        latest_s,
        model.predict_proba,
        num_features=len(features),
        num_samples=500
    )

    # Build a clean matplotlib figure from LIME output
    lime_list = exp.as_list()
    labels = [x[0] for x in lime_list]
    vals = [x[1] for x in lime_list]
    colors = ["#C9A84C" if v > 0 else "#8B0000" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = range(len(labels))
    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="white", linewidth=0.8)
    ax.set_xlabel("LIME Weight (positive = supports UP)", fontsize=11)
    ax.set_title("LIME - Why This Specific Prediction Was Made", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(colors="white")
    plt.tight_layout()

    return {"fig": fig, "explanation": exp, "lime_list": lime_list}


# ─── SAVE / LOAD ─────────────────────────────────────────────────────────────

def save_artifacts(artifacts: dict, symbol: str):
    path = os.path.join(MODEL_DIR, f"{symbol.replace('.', '_')}_artifacts.joblib")
    joblib.dump({k: v for k, v in artifacts.items() if k not in ["X_train", "X_test", "X_train_s", "X_test_s"]}, path)


def load_artifacts(symbol: str) -> dict:
    path = os.path.join(MODEL_DIR, f"{symbol.replace('.', '_')}_artifacts.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None
