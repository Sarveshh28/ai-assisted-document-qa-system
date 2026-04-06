from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, MODEL_BUNDLE_PATH, MODEL_FEATURES, NUMERIC_FEATURES
from src.pipeline.features import validate_input_frame


def build_preprocessor() -> ColumnTransformer:
    derived = ["thermal_stress", "mechanical_stress", "energy_intensity", "maintenance_urgency"]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES + derived),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def train_models(frame: pd.DataFrame) -> tuple[dict, dict]:
    data = validate_input_frame(frame)
    X = data[MODEL_FEATURES]
    y = data["failure_within_7_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifier = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=220,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)[:, 1]

    anomaly_features = data[NUMERIC_FEATURES + ["thermal_stress", "mechanical_stress", "energy_intensity", "maintenance_urgency"]]
    anomaly_scaler = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    anomaly_matrix = anomaly_scaler.fit_transform(anomaly_features)
    anomaly_model = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
    anomaly_model.fit(anomaly_matrix)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_score": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    bundle = {
        "classifier": classifier,
        "anomaly_scaler": anomaly_scaler,
        "anomaly_model": anomaly_model,
        "model_features": MODEL_FEATURES,
        "numeric_anomaly_features": NUMERIC_FEATURES + ["thermal_stress", "mechanical_stress", "energy_intensity", "maintenance_urgency"],
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    return bundle, metrics
