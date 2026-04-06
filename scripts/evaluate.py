import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.engine import ResearchAssistantEngine
from src.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSON file with evaluation examples")
    parser.add_argument("--top-k", type=int, default=5, help="Number of passages to retrieve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = read_json(args.dataset)
    engine = ResearchAssistantEngine()
    metrics = engine.evaluate(dataset, top_k=args.top_k, save_report=True)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
