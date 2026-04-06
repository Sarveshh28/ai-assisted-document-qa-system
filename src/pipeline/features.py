from __future__ import annotations

import pandas as pd


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["thermal_stress"] = (data["temperature"] - 68) * 0.05 + (data["humidity"] - 50) * 0.01
    data["mechanical_stress"] = data["vibration"] * 0.55 + (data["rpm"] - 1400) / 700 + data["tool_wear"] * 0.02
    data["energy_intensity"] = data["power_draw"] / data["rpm"].clip(lower=1)
    data["maintenance_urgency"] = data["maintenance_gap_days"] * 0.04 + data["operating_hours"] * 0.06 + data["age_days"] / 1200
    return data


def validate_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "machine_type",
        "temperature",
        "vibration",
        "pressure",
        "rpm",
        "humidity",
        "power_draw",
        "operating_hours",
        "maintenance_gap_days",
        "tool_wear",
        "age_days",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return add_derived_features(frame)
