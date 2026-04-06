import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.indexer import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the retrieval index from text documents.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .txt files")
    parser.add_argument("--chunk-size", type=int, default=120, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=30, help="Chunk overlap in words")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_index(args.input_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    print("Index build complete")
    print(summary)


if __name__ == "__main__":
    main()
