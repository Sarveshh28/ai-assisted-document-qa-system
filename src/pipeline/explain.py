from __future__ import annotations


SAFE_RANGES = {
    "temperature": (55, 85),
    "vibration": (2, 6),
    "pressure": (24, 38),
    "rpm": (1200, 1650),
    "humidity": (35, 65),
    "power_draw": (220, 380),
    "operating_hours": (6, 18),
    "maintenance_gap_days": (5, 35),
    "tool_wear": (10, 65),
    "age_days": (120, 850),
}


def _distance_from_range(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        midpoint = (low + high) / 2
        span = max(high - low, 1)
        return abs(value - midpoint) / span
    if value < low:
        return (low - value) / max(high - low, 1) + 1
    return (value - high) / max(high - low, 1) + 1


def explain_record(record: dict) -> list[dict]:
    scored = []
    for feature, (low, high) in SAFE_RANGES.items():
        value = float(record[feature])
        distance = _distance_from_range(value, low, high)
        scored.append({"feature": feature, "value": round(value, 3), "impact": round(distance, 3)})
    return sorted(scored, key=lambda item: item["impact"], reverse=True)[:4]


def recommendation_from_risk(probability: float, anomaly_flag: bool) -> str:
    if probability >= 0.8 or anomaly_flag:
        return "Immediate inspection recommended. Stop the machine if abnormal behavior persists."
    if probability >= 0.55:
        return "Schedule maintenance within 24 to 48 hours and monitor sensor drift closely."
    if probability >= 0.35:
        return "Machine is stable but should be monitored during the next maintenance cycle."
    return "Machine condition looks healthy. Continue normal operation and routine checks."


def risk_level(probability: float, anomaly_flag: bool) -> str:
    adjusted = probability + (0.15 if anomaly_flag else 0.0)
    if adjusted >= 0.8:
        return "critical"
    if adjusted >= 0.55:
        return "high"
    if adjusted >= 0.3:
        return "moderate"
    return "low"
