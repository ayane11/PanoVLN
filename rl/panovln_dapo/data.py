"""Lossless episode manifests: keep annotation metadata required by Habitat.

This module is CPU-only. Parquet support is imported only when writing a file.
Each row retains its dataset metadata, so filtering/sharding cannot detach an
R2R episode from instruction_vocab. The extra duplication compresses in Parquet.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def episode_json(episode, *, allow_rxr_timing_nan=False):
    """Preserve source RxR NaN timestamps in the embedded JSON string only.

    Python/Habitat accept these source JSON NaNs. No nonfinite values are
    permitted in geometry or other metadata. Parquet columns remain finite.
    """
    allowed = {('instruction','timed_instruction','[]','start_time'),
               ('instruction','timed_instruction','[]','end_time')}
    def check(value, path=()):
        if isinstance(value,float) and not math.isfinite(value):
            if not (allow_rxr_timing_nan and math.isnan(value) and path in allowed):
                raise ValueError('Nonfinite annotation field: '+'.'.join(path))
        elif isinstance(value,dict):
            for key,child in value.items(): check(child,path+(key,))
        elif isinstance(value,list):
            for child in value: check(child,path+('[]',))
    check(episode)
    return json.dumps(episode,ensure_ascii=False,sort_keys=True,separators=(',', ':'),
                      allow_nan=allow_rxr_timing_nan)


def validate_episode(episode, *, allow_rxr_timing_nan=False):
    if not isinstance(episode, dict):
        raise ValueError("Episode must be an object")
    for key in ("scene_id", "episode_id"):
        if key not in episode or episode[key] is None or not str(episode[key]).strip():
            raise ValueError(f"Missing {key}")
    instruction = episode.get("instruction", {})
    if not isinstance(instruction, dict) or not isinstance(instruction.get("instruction_text"), str) or not instruction["instruction_text"].strip():
        raise ValueError("Expected nonempty instruction.instruction_text")

    def vector(value, length):
        return (isinstance(value, list) and len(value) == length and
                all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value))

    if not vector(episode.get("start_position"), 3) or not vector(episode.get("start_rotation"), 4):
        raise ValueError("Invalid start pose")
    norm = sum(x*x for x in episode["start_rotation"])
    if abs(norm - 1) > 1e-3:
        raise ValueError("Start rotation must be a unit quaternion")
    goals = episode.get("goals")
    if not isinstance(goals, list) or not goals or any(not isinstance(g, dict) or not vector(g.get("position"), 3) for g in goals):
        raise ValueError("Expected nonempty goals with finite positions")
    path = episode.get("reference_path")
    if not isinstance(path, list) or not path or any(not vector(point, 3) for point in path):
        raise ValueError("Expected a finite nonempty reference_path")
    episode_json(episode,allow_rxr_timing_nan=allow_rxr_timing_nan)


def annotation_rows(path, *, dataset_name, split, dataset_type, languages=None):
    if dataset_type not in ("R2RVLNCE-v1", "RxRVLNCE-v1"):
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    if not dataset_name or not split:
        raise ValueError("Dataset name and split are required")
    if languages is not None and (dataset_type != 'RxRVLNCE-v1' or not languages):
        raise ValueError('Language selection requires RxR and nonempty languages')
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    metadata = {k: v for k, v in payload.items() if k != "episodes"}
    if dataset_type == "R2RVLNCE-v1":
        vocab = metadata.get("instruction_vocab", {}).get("word_list")
        if not isinstance(vocab, list):
            raise ValueError("R2R requires instruction_vocab.word_list")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Annotation must contain episodes")
    source_sha = hashlib.sha256(raw).hexdigest()
    metadata_json = canonical(metadata)
    rows, seen = [], set()
    for episode in episodes:
        if languages is not None and episode.get('instruction',{}).get('language') not in languages:
            continue
        validate_episode(episode,allow_rxr_timing_nan=dataset_type=='RxRVLNCE-v1')
        serialized=episode_json(episode,allow_rxr_timing_nan=dataset_type=='RxRVLNCE-v1')
        instruction = episode["instruction"]["instruction_text"]
        key = digest([dataset_name, split, episode["scene_id"], str(episode["episode_id"]), instruction])
        if key in seen:
            raise ValueError(f"Duplicate episode identity: {key}")
        seen.add(key)
        rows.append({
            "data_source": "vln",
            "prompt": [{"role": "user", "content": "placeholder"}],
            "agent_name": "panovln_full_episode_agent",
            "extra_info": {"episode_id": str(episode["episode_id"]), "scene_id": episode["scene_id"],
                           "instruction": instruction, "dataset_name": dataset_name,
                           "split": split, "episode_key": key},
            "episode": serialized, "episode_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "dataset_type": dataset_type, "annotation_metadata": metadata_json,
            "source_sha256": source_sha,
        })
    if not rows: raise ValueError('No episodes match the language selection')
    return rows


def restore_annotation(rows):
    """Restore one source for its native loader; refuse ambiguous mixed sources."""
    if not rows:
        raise ValueError("Cannot restore an empty manifest")
    first = rows[0]
    signature = tuple(first[k] for k in ("dataset_type", "annotation_metadata", "source_sha256"))
    seen, episodes = set(), []
    for row in rows:
        if tuple(row[k] for k in ("dataset_type", "annotation_metadata", "source_sha256")) != signature:
            raise ValueError("Partition mixed manifests by source before Habitat loading")
        episode = json.loads(row["episode"])
        validate_episode(episode,allow_rxr_timing_nan=row['dataset_type']=='RxRVLNCE-v1')
        serialized=episode_json(episode,allow_rxr_timing_nan=row['dataset_type']=='RxRVLNCE-v1')
        extra = row["extra_info"]
        key = digest([extra["dataset_name"], extra["split"], episode["scene_id"], str(episode["episode_id"]), episode["instruction"]["instruction_text"]])
        if hashlib.sha256(serialized.encode()).hexdigest() != row["episode_sha256"] or key != extra["episode_key"]:
            raise ValueError("Episode integrity mismatch")
        if (extra["scene_id"], extra["episode_id"], extra["instruction"]) != (episode["scene_id"], str(episode["episode_id"]), episode["instruction"]["instruction_text"]):
            raise ValueError("Episode routing mismatch")
        if key in seen:
            raise ValueError("Duplicate manifest row")
        seen.add(key)
        episodes.append(episode)
    return {**json.loads(first["annotation_metadata"]), "episodes": episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset-type", choices=["R2RVLNCE-v1", "RxRVLNCE-v1"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", help="RxR instruction languages, e.g. en-US en-IN; default keeps all")
    args = parser.parse_args()
    rows = annotation_rows(args.annotation, dataset_name=args.dataset_name, split=args.split, dataset_type=args.dataset_type, languages=args.languages)
    restore_annotation(rows)
    import pyarrow as pa
    import pyarrow.parquet as pq
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(f"Wrote {len(rows)} episodes to {args.output}")


if __name__ == "__main__":
    main()
