from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sensor_data(rows: int = 3000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    machine_types = rng.choice(["cnc", "compressor", "robotic_arm", "turbine"], size=rows, p=[0.3, 0.25, 0.2, 0.25])
    temperature = rng.normal(74, 10, rows)
    vibration = rng.normal(4.2, 1.4, rows)
    pressure = rng.normal(31, 6, rows)
    rpm = rng.normal(1450, 180, rows)
    humidity = rng.normal(52, 12, rows)
    power_draw = rng.normal(305, 65, rows)
    operating_hours = rng.uniform(4, 24, rows)
    maintenance_gap_days = rng.integers(2, 60, rows)
    tool_wear = rng.uniform(5, 100, rows)
    age_days = rng.integers(90, 1200, rows)

    type_bias = np.select(
        [machine_types == "cnc", machine_types == "compressor", machine_types == "robotic_arm", machine_types == "turbine"],
        [0.1, 0.18, 0.08, 0.22],
        default=0.0,
    )

    thermal_stress = (temperature - 68) * 0.05 + (humidity - 50) * 0.01
    mechanical_stress = vibration * 0.55 + (rpm - 1400) / 700 + tool_wear * 0.02
    energy_intensity = power_draw / np.maximum(rpm, 1)
    maintenance_urgency = maintenance_gap_days * 0.04 + operating_hours * 0.06 + age_days / 1200

    risk_signal = (
        0.9 * thermal_stress
        + 1.4 * mechanical_stress
        + 8.0 * energy_intensity
        + 0.8 * maintenance_urgency
        + type_bias
        + rng.normal(0, 0.55, rows)
    )

    failure_score = 1 / (1 + np.exp(-(risk_signal - 8.8)))
    failure_within_7_days = (failure_score > 0.65).astype(int)
    anomaly = ((vibration > 7.0) | (temperature > 95) | (pressure < 18) | (power_draw > 430)).astype(int)

    return pd.DataFrame(
        {
            "machine_id": [f"M-{idx:05d}" for idx in range(1, rows + 1)],
            "machine_type": machine_types,
            "temperature": np.round(temperature, 2),
            "vibration": np.round(vibration, 2),
            "pressure": np.round(pressure, 2),
            "rpm": np.round(rpm, 2),
            "humidity": np.round(humidity, 2),
            "power_draw": np.round(power_draw, 2),
            "operating_hours": np.round(operating_hours, 2),
            "maintenance_gap_days": maintenance_gap_days,
            "tool_wear": np.round(tool_wear, 2),
            "age_days": age_days,
            "thermal_stress": np.round(thermal_stress, 3),
            "mechanical_stress": np.round(mechanical_stress, 3),
            "energy_intensity": np.round(energy_intensity, 3),
            "maintenance_urgency": np.round(maintenance_urgency, 3),
            "anomaly_label": anomaly,
            "failure_within_7_days": failure_within_7_days,
        }
    )
