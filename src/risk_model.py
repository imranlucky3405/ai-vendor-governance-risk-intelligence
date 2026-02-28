import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

FEATURES = [
    "avg_delay_days",
    "delay_frequency",
    "avg_payment_delay",
    "payment_risk_ratio",
    "rejection_rate_pct",
    "penalty_cases",
    "relationship_years",
    "contract_value_lakhs",
]

def train_and_save_model(data_path: str, model_path: str = "models/risk_model.pkl"):
    df = pd.read_csv(data_path)

    # Low VPI = High Risk (target = 1)
    df["risk_target"] = (df["VPI_Category"] == "Low").astype(int)

    missing = sorted(set(FEATURES + ["VPI_Category"]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES].copy()
    y = df["risk_target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "label_definition": "1 = Low VPI (High Risk)"
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, model_path)

    print("Saved:", model_path)
    print("ROC-AUC:", metrics["roc_auc"])
    return model

def load_model(model_path: str = "models/risk_model.pkl"):
    return joblib.load(model_path)