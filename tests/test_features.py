import pandas as pd

from src.pipeline.features import add_derived_features


def test_add_derived_features_adds_expected_columns():
    frame = pd.DataFrame(
        [
            {
                "machine_type": "cnc",
                "temperature": 80.0,
                "vibration": 5.0,
                "pressure": 30.0,
                "rpm": 1400.0,
                "humidity": 52.0,
                "power_draw": 300.0,
                "operating_hours": 12.0,
                "maintenance_gap_days": 20,
                "tool_wear": 50.0,
                "age_days": 400,
            }
        ]
    )
    result = add_derived_features(frame)
    assert "thermal_stress" in result.columns
    assert "mechanical_stress" in result.columns
