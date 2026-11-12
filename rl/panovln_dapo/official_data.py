"""Import the released SimpleVLN manifest without altering episode semantics."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .data import canonical, digest, episode_json, restore_annotation, validate_episode

OFFICIAL_SHA256 = '53709e53f7a22cbd9439883337c9484ffef16d8fcc2093e5a13bb0e341ff0a9c'


def convert_rows(source_rows, source_sha):
    # Pano consumes instruction_text with its own model tokenizer. Habitat's
    # legacy word vocabulary is a loader requirement, not a model tokenizer.
    metadata = canonical({'instruction_vocab': {'word_list': []}})
    rows = []
    for source in source_rows:
        episode = json.loads(source['episode'])
        validate_episode(episode)
        extra = source['extra_info']
        instruction = episode['instruction']['instruction_text']
        if (str(extra['episode_id']) != str(episode['episode_id']) or
                extra['scene_id'] != episode['scene_id'] or
                extra['instruction'].strip() != instruction.strip()):
            raise ValueError('Official manifest routing disagrees with embedded episode')
        path = Path(episode['scene_id'])
        if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('mp3d', 'hm3d'):
            raise ValueError('Invalid portable scene path')
        serialized = episode_json(episode)
        key = digest(['simplevln_official', 'train', episode['scene_id'],
                      str(episode['episode_id']), instruction])
        rows.append(dict(data_source='vln', prompt=source['prompt'],
                         agent_name='panovln_full_episode_agent',
                         extra_info=dict(episode_id=str(episode['episode_id']),
                                         scene_id=episode['scene_id'], instruction=instruction,
                                         dataset_name='simplevln_official', split='train', episode_key=key),
                         episode=serialized, episode_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                         dataset_type='R2RVLNCE-v1', annotation_metadata=metadata,
                         source_sha256=source_sha))
    restored = restore_annotation(rows)
    if restored['episodes'] != [json.loads(r['episode']) for r in source_rows]:
        raise ValueError('Episode round trip changed source data')
    return rows, restored


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if sha != OFFICIAL_SHA256:
        raise ValueError('Unexpected official release checksum; audit new release before importing')
    sources = pq.read_table(args.source).to_pylist()
    if len(sources) != 6901:
        raise ValueError('Expected the audited 6901-task release')
    rows, annotation = convert_rows(sources, sha)
    args.output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(rows), args.output/'train.parquet')
    with gzip.open(args.output/'train.json.gz', 'wt') as stream:
        json.dump(annotation, stream, ensure_ascii=False, allow_nan=False)
    report = dict(source=str(args.source), source_sha256=sha, rows=len(rows),
                  scenes=len({r['extra_info']['scene_id'] for r in rows}),
                  lossless_episode_roundtrip=True, instruction_sensor_enabled=False,
                  vocabulary_note='Empty legacy vocabulary; raw instruction_text uses native Pano tokenizer')
    (args.output/'import_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
