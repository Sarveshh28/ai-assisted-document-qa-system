from __future__ import annotations

import joblib
import pandas as pd

from src.config import MODEL_BUNDLE_PATH
from src.pipeline.explain import explain_record, recommendation_from_risk, risk_level
from src.pipeline.features import validate_input_frame


def load_bundle() -> dict:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run `python scripts/train.py --data data/raw/machine_sensor_data.csv` first.")
    return joblib.load(MODEL_BUNDLE_PATH)


def predict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    bundle = load_bundle()
    data = validate_input_frame(frame)

    classifier = bundle["classifier"]
    anomaly_scaler = bundle["anomaly_scaler"]
    anomaly_model = bundle["anomaly_model"]
    numeric_features = bundle["numeric_anomaly_features"]

    failure_probability = classifier.predict_proba(data[bundle["model_features"]])[:, 1]
    predicted_failure = classifier.predict(data[bundle["model_features"]])
    transformed = anomaly_scaler.transform(data[numeric_features])
    anomaly_scores = anomaly_model.decision_function(transformed)
    anomaly_flags = anomaly_model.predict(transformed) == -1

    records = []
    for position, (_, row) in enumerate(data.iterrows()):
        probability = float(failure_probability[position])
        anomaly_flag = bool(anomaly_flags[position])
        current = row.to_dict()
        current["failure_probability"] = round(probability, 4)
        current["predicted_failure"] = int(predicted_failure[position])
        current["anomaly_score"] = round(float(anomaly_scores[position]), 4)
        current["anomaly_flag"] = anomaly_flag
        current["risk_level"] = risk_level(probability, anomaly_flag)
        current["top_factors"] = explain_record(row.to_dict())
        current["recommendation"] = recommendation_from_risk(probability, anomaly_flag)
        records.append(current)
    return pd.DataFrame(records)
