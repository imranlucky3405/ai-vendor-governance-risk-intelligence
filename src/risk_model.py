import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_and_save_model(data_path, model_path="models/risk_model.pkl"):

    # Load dataset
    df = pd.read_csv(data_path)

    # Create target
    df["risk_target"] = (df["VPI_Category"] == "Low").astype(int)

    features = [
        "avg_delay_days",
        "delay_frequency",
        "avg_payment_delay",
        "payment_risk_ratio",
        "rejection_rate_pct",
        "penalty_cases",
        "relationship_years",
        "contract_value_lakhs"
    ]

    X = df[features]
    y = df["risk_target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Model Performance")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    # Save model
    joblib.dump({
        "model": model,
        "features": features
    }, model_path)

    print("Model saved at:", model_path)

    return model


def load_model(model_path="models/risk_model.pkl"):
    return joblib.load(model_path)