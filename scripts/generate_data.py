import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_PATH, RAW_DIR
from src.pipeline.data_generator import generate_sensor_data
from src.utils.io import ensure_dir, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic machine sensor data.")
    parser.add_argument("--rows", type=int, default=3000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(RAW_DIR)
    frame = generate_sensor_data(rows=args.rows, random_state=args.seed)
    write_csv(DATASET_PATH, frame)
    print(f"Generated dataset at {DATASET_PATH}")
    print(frame.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
