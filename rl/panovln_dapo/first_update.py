"""Supervise the first real-reward update after an already-running pilot exits.

Only the explicitly identified pilot service is stopped. GPU occupancy, sampler
completion and batch integrity must pass before launching the two-rank actor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def identity(pid):
    p=Path(f'/proc/{pid}')
    try:
        stat=(p/'stat').read_text().split(') ',1)[1].split()
        if stat[0]=='Z':return None
        return stat[19],(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',required=True,type=Path)
    p.add_argument('--collector-pid',required=True,type=int)
    p.add_argument('--service-pid',required=True,type=int)
    p.add_argument('--service-port',required=True,type=int)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def status(stage,**kwargs):
        record=dict(stage=stage,time=time.time(),**kwargs)
        (args.output/'status.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    try:
        collector=identity(args.collector_pid)
        service=identity(args.service_pid)
        if collector is not None and 'panovln_dapo.collect_pilot' not in collector[1]:
            raise ValueError('Collector PID does not identify this pilot')
        if service is None or 'panovln_dapo.env_server' not in service[1] or f'--port {args.service_port}' not in service[1]:
            raise ValueError('Service PID/port identity differs')
        status('waiting_for_pilot',collector_pid=args.collector_pid,service_pid=args.service_pid)
        while collector is not None:
            current=identity(args.collector_pid)
            if current is None:break
            if current!=collector:raise RuntimeError('Collector PID was reused')
            time.sleep(10)
        if (args.pilot/'failure.json').exists() or not (args.pilot/'complete.json').exists():
            raise RuntimeError('Pilot exited without successful completion')
        from .actor import load_batch
        batch=args.pilot/'selected_batch.json'
        payload,rows=load_batch(batch)
        status('batch_validated',decisions=len(rows),trajectories=len(payload['trajectories']),metrics=payload['metrics'])
        checkpoint=payload['checkpoint']
        del payload,rows
        with urllib.request.urlopen(f'http://127.0.0.1:{args.service_port}/stats',timeout=10) as response:
            stats=json.load(response)
        if stats['active_sessions']!=0:raise RuntimeError('Pilot service still has active sessions')
        if identity(args.service_pid)!=service:raise RuntimeError('Service identity changed before shutdown')
        os.kill(args.service_pid,signal.SIGTERM)
        deadline=time.monotonic()+60
        while identity(args.service_pid) is not None and time.monotonic()<deadline:time.sleep(1)
        if identity(args.service_pid) is not None:raise RuntimeError('Service did not exit after SIGTERM')
        time.sleep(3)
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
        if any(memory[g]>1000 for g in (3,6)):raise RuntimeError(f'GPU 3/6 not idle: {memory}')
        snapshot=args.output/'source_snapshot';snapshot.mkdir()
        hashes={}
        for source in Path('rl/panovln_dapo').glob('*.py'):
            data=source.read_bytes();(snapshot/source.name).write_bytes(data)
            hashes[source.name]=hashlib.sha256(data).hexdigest()
        import sys
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                 '-m','panovln_dapo.actor','--batch',str(batch),'--checkpoint',checkpoint,
                 '--output',str(args.output/'checkpoint'),'--lr','5e-7','--ppo-epochs','1','--mini-batch-decisions','128']
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3,6',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        (args.output/'run_manifest.json').write_text(json.dumps(dict(command=command,gpus=[3,6],
            source_sha256=hashes,batch_sha256=hashlib.sha256(batch.read_bytes()).hexdigest(),
            service_shutdown=stats),indent=2)+'\n')
        with (args.output/'actor.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            status('actor_running',pid=child.pid)
            code=child.wait()
        if code:raise RuntimeError(f'Actor exited with status {code}; inspect actor.log')
        if not (args.output/'checkpoint/vln_progress.json').exists():raise RuntimeError('Actor returned without checkpoint metadata')
        status('update_exported',checkpoint=str(args.output/'checkpoint'))
    except BaseException as exc:
        status('failed',error=str(exc))
        raise


if __name__=='__main__':main()
