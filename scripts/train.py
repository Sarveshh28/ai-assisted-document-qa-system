import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_PATH, MODELS_DIR, REPORTS_DIR
from src.pipeline.trainer import train_models
from src.utils.io import ensure_dir, read_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train predictive maintenance models.")
    parser.add_argument("--data", type=Path, required=True, help="Path to training CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)
    frame = read_csv(args.data)
    _, metrics = train_models(frame)
    write_json(METRICS_PATH, metrics)
    print("Training complete")
    print(metrics)


if __name__ == "__main__":
    main()
