from src.pipeline.explain import risk_level


def test_risk_level_escalates_for_anomaly():
    assert risk_level(0.7, True) == "critical"
