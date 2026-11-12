"""Locate the current SimpleVLN checkout, never historical copies."""
from pathlib import Path
import sys


def add_source_path():
    root = Path(__file__).resolve().parents[3] / 'SimpleVLN' / 'RL'
    if not (root / 'recipe/vln_simplevln/env_server/server.py').is_file():
        raise FileNotFoundError(f'SimpleVLN RL source not found: {root}')
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    return root
