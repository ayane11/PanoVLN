"""Exercise native ERP rendering and metrics for each scene and flagged start."""
import argparse
import json
import math
from pathlib import Path
import time


def main():
    import torch
    import numpy as np
    from PIL import Image
    from .simplevln import add_source_path
    add_source_path()
    from .env_server import load_pano_env
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--assets-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    asset_report = json.loads(args.assets_audit.read_text())
    if not asset_report['passed']:
        raise ValueError('Assets gate failed')
    warnings = {(r['scene'], str(r['episode_id'])) for r in asset_report['warnings']}
    env, full, _ = load_pano_env(args.config)
    seen, selected, failures = set(), [], []
    representatives = set()
    started = time.time()
    try:
        for episodes in full.values():
            for ep in episodes:
                if ep.scene_id not in seen or (ep.scene_id, str(ep.episode_id)) in warnings:
                    selected.append(ep)
                seen.add(ep.scene_id)
        for i, ep in enumerate(selected, 1):
            record = dict(scene=ep.scene_id, episode_id=str(ep.episode_id))
            try:
                env.current_episode = ep
                obs = env.reset()
                rgb = obs['rgb']
                if rgb.shape != (640, 1280, 3) or not np.isfinite(rgb).all() or float(rgb.std()) < 1:
                    raise ValueError('Invalid/blank ERP frame: '+str(rgb.shape))
                before = env.get_metrics()
                if not math.isfinite(before['distance_to_goal']):
                    raise ValueError('Nonfinite native distance metric')
                info = ep.info or {}
                category = ('hm3d' if '/hm3d/' in ep.scene_id else
                            'rxr' if info.get('role') == 'guide' else
                            'mp3d_other' if 'rooms' in info else 'r2r')
                if category not in representatives:
                    Image.fromarray(rgb).save(args.output/(category+'.png'))
                    (args.output/(category+'.json')).write_text(json.dumps(dict(instruction=ep.instruction.instruction_text, scene=ep.scene_id, episode_id=str(ep.episode_id)), indent=2)+'\n')
                    representatives.add(category)
                env.step(2)  # Native TURN_LEFT, then STOP.
                env.step(0)
                metrics = env.get_metrics()
                for name in ('distance_to_goal', 'success', 'spl', 'oracle_success'):
                    if not math.isfinite(metrics[name]):
                        raise ValueError('Nonfinite '+name)
                record.update(metrics=metrics, rgb_std=float(rgb.std()), passed=True)
            except Exception as exc:
                record.update(passed=False, error=str(exc)); failures.append(record)
            with (args.output/'episodes.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            if i % 10 == 0:
                print(json.dumps(dict(checked=i,total=len(selected),failures=len(failures),elapsed=time.time()-started)),flush=True)
        report = dict(passed=not failures, scenes=len(seen), episodes_checked=len(selected),
                      flagged_starts_checked=len(warnings), failures=failures,
                      representative_categories=sorted(representatives), elapsed=time.time()-started)
        (args.output/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
        if failures: raise RuntimeError('Render/metric compatibility gate failed')
    finally:
        env.close()


if __name__ == '__main__':
    main()
