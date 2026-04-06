import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.engine import ResearchAssistantEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the Research Assistant from the CLI.")
    parser.add_argument("--question", type=str, required=True, help="User question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of passages to retrieve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = ResearchAssistantEngine()
    result = engine.ask(args.question, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
