"""Check every released scene/navmesh and every episode's route on CPU."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import struct
import time


def main():
    import torch  # Habitat's native import order.
    import habitat_sim
    from habitat_extensions.task import VLNDatasetV1
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotation', type=Path, required=True)
    p.add_argument('--scenes-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.annotation, 'rt') as f:
        payload = json.load(f)
    native = VLNDatasetV1()
    native.from_json(json.dumps(payload), scenes_dir=str(args.scenes_root))
    if len(native.episodes) != len(payload['episodes']):
        raise ValueError('Native loader dropped episodes')
    grouped = defaultdict(list)
    for episode in native.episodes:
        grouped[episode.scene_id].append(episode)
    failures, warnings = [], []
    started = time.time()
    for number, (scene, episodes) in enumerate(sorted(grouped.items()), 1):
        record = dict(scene=scene, episodes=len(episodes))
        try:
            model = Path(scene)
            with model.open('rb') as f:
                magic, version, length = struct.unpack('<4sII', f.read(12))
            if magic != b'glTF' or version != 2 or length != model.stat().st_size:
                raise ValueError('Invalid or incomplete GLB')
            nav = model.with_suffix('.navmesh')
            pf = habitat_sim.PathFinder()
            if not pf.load_nav_mesh(str(nav)) or not pf.is_loaded:
                raise ValueError('Cannot load navmesh: '+str(nav))
            if not math.isfinite(pf.navigable_area) or pf.navigable_area <= 0:
                raise ValueError('Empty navigation area')
            record.update(model_bytes=length, navmesh_bytes=nav.stat().st_size,
                          navigable_area=pf.navigable_area)
            for ep in episodes:
                distances = []
                for goal in ep.goals:
                    route = habitat_sim.ShortestPath()
                    route.requested_start = ep.start_position
                    route.requested_end = goal.position
                    if pf.find_path(route) and math.isfinite(route.geodesic_distance):
                        distances.append(route.geodesic_distance)
                if not distances:
                    failures.append(dict(scene=scene, episode_id=ep.episode_id, error='No finite start-to-goal path'))
                if not pf.is_navigable(ep.start_position):
                    warnings.append(dict(scene=scene, episode_id=ep.episode_id, warning='Start not on navmesh within default tolerance'))
        except Exception as exc:
            failures.append(dict(scene=scene, error=str(exc)))
            record['error'] = str(exc)
        with (args.output/'scenes.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if number % 25 == 0:
            print(json.dumps(dict(scenes_checked=number, total=len(grouped), failures=len(failures), elapsed=time.time()-started)), flush=True)
    report = dict(episodes=len(native.episodes), scenes=len(grouped), native_loader='R2RVLNCE-v1',
                  failures=failures, warnings=warnings, elapsed_seconds=time.time()-started,
                  passed=not failures)
    (args.output/'complete.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('failures','warnings')}), flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} asset/route failures; inspect complete.json')


if __name__ == '__main__':
    main()
