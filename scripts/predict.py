import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.predictor import predict_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict machine health from a single reading.")
    parser.add_argument("--machine-type", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--vibration", type=float, required=True)
    parser.add_argument("--pressure", type=float, required=True)
    parser.add_argument("--rpm", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--power-draw", type=float, required=True)
    parser.add_argument("--operating-hours", type=float, required=True)
    parser.add_argument("--maintenance-gap-days", type=int, required=True)
    parser.add_argument("--tool-wear", type=float, required=True)
    parser.add_argument("--age-days", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.DataFrame(
        [
            {
                "machine_type": args.machine_type,
                "temperature": args.temperature,
                "vibration": args.vibration,
                "pressure": args.pressure,
                "rpm": args.rpm,
                "humidity": args.humidity,
                "power_draw": args.power_draw,
                "operating_hours": args.operating_hours,
                "maintenance_gap_days": args.maintenance_gap_days,
                "tool_wear": args.tool_wear,
                "age_days": args.age_days,
            }
        ]
    )
    result = predict_frame(frame).iloc[0].to_dict()
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
