"""Audit training data and persist an offset-only index, without using GPUs."""
import argparse
import json
from pathlib import Path
from src.memory.stream import build_episode_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-summaries', action='store_true')
    parser.add_argument('--sub-summaries')
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    index = build_episode_index(args.jsonl, not args.no_summaries, args.sub_summaries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(index, handle)
    print(f"Indexed {len(index['episodes'])} episodes in {path}")


if __name__ == '__main__':
    main()
