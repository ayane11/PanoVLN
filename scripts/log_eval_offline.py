"""Record completed native evaluation metrics to a local offline W&B run."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--eval-config', required=True)
    parser.add_argument('--script', required=True, type=Path)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    metrics = json.loads((args.results / 'result_summary.json').read_text())
    os.environ['WANDB_MODE'] = 'offline'
    import wandb
    args.results.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        project=os.environ.get('WANDB_PROJECT', 'panovln-eval'),
        name=f'{args.results.parent.name}-{args.results.name}',
        mode='offline', dir=str(args.results.resolve()), job_type='evaluation',
        config={'model_path': str(Path(args.model).resolve()),
                'eval_config': args.eval_config, 'gpu': args.gpu,
                'eval_script': str(args.script.resolve()),
                'eval_script_sha256': hashlib.sha256(args.script.read_bytes()).hexdigest()},
    ) as run:
        values={f'eval/{k}': v for k,v in metrics.items() if isinstance(v, (int,float))}
        run.log(values)
        run.summary.update(values)
        print(f'W&B offline evaluation saved: {run.dir}', flush=True)


if __name__ == '__main__':
    main()
