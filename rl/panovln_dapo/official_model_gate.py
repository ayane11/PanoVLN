"""Check native ERP inputs and generation/replay probabilities before training."""
import argparse
import base64
import io
import json
from pathlib import Path


def main():
    import torch
    from PIL import Image
    from .simplevln import add_source_path
    from .backend import NativePolicy, response_logprobs
    add_source_path()
    from .env_server import load_pano_env
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # CUDA_VISIBLE_DEVICES=rollout,environment; never use a third device.
    config = Path(args.config).read_text().replace('gpu_device_id: 0', 'gpu_device_id: 1')
    path = args.output/'environment.yaml'; path.write_text(config)
    env, full, _ = load_pano_env(str(path.resolve()))
    selected = {}
    for episodes in full.values():
        for ep in episodes:
            info = ep.info or {}
            category = ('hm3d' if '/hm3d/' in ep.scene_id else 'rxr' if info.get('role') == 'guide'
                        else 'mp3d_other' if 'rooms' in info else 'r2r')
            if category not in selected or len(ep.instruction.instruction_text) > len(selected[category].instruction.instruction_text):
                selected[category] = ep
    if set(selected) != {'hm3d', 'rxr', 'mp3d_other', 'r2r'}:
        raise ValueError('Missing official source category')
    frames = {}
    try:
        for category, ep in selected.items():
            env.current_episode = ep
            rgb = env.reset()['rgb']
            image = Image.fromarray(rgb)
            image.save(args.output/(category+'.png'))
            buffer = io.BytesIO(); image.save(buffer, format='JPEG')
            frames[category] = base64.b64encode(buffer.getvalue()).decode()
    finally:
        env.close()
    torch.set_num_threads(2); torch.manual_seed(42)
    policy = NativePolicy.load(args.checkpoint, dtype=torch.float32, attention='sdpa', max_response_tokens=64)
    reports = []
    for category, ep in selected.items():
        # Exercise full history on the longest source instructions, with native
        # Pano geometry and frozen VGGT enabled. These are probes, not rollouts.
        history = [frames[category]] * 11
        gen = policy.generate(ep.instruction.instruction_text, history)
        inputs = policy.prepare(ep.instruction.instruction_text, history)['inputs']
        with torch.no_grad():
            replay = response_logprobs(policy.model, inputs, gen.response_ids)
        error = (replay-torch.tensor(gen.response_logprobs,device=replay.device)).abs()
        record = dict(category=category, episode_id=str(ep.episode_id), scene=ep.scene_id,
                      history_frames=11, prompt_tokens=len(gen.prompt_ids), response_tokens=len(gen.response_ids),
                      response=gen.action_text, max_error=float(error.max()), mean_error=float(error.mean()),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        reports.append(record); print(json.dumps(record), flush=True)
        if not torch.isfinite(replay).all() or error.max()>1e-3 or error.mean()>1e-4:
            raise ValueError('Native generation/replay probability gate failed')
    (args.output/'complete.json').write_text(json.dumps(dict(passed=True,reports=reports),indent=2)+'\n')


if __name__ == '__main__':
    main()
