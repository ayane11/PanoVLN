"""After a live actor exits successfully, audit export and run paired evaluation."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from .first_update import identity


def update_initial_checkpoint(actor_run):
    """Audit the consecutive update; evaluation baseline stays fixed separately."""
    actor_run=Path(actor_run)
    command=json.loads((actor_run/'run_manifest.json').read_text())['command']
    if command.count('--checkpoint')!=1 or command.count('--output')!=1:
        raise ValueError('Actor manifest lacks unambiguous checkpoint/output arguments')
    initial=Path(command[command.index('--checkpoint')+1])
    output=Path(command[command.index('--output')+1])
    if output.resolve()!=(actor_run/'checkpoint').resolve():
        raise ValueError('Actor manifest export path differs from this run')
    return initial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor-run',required=True,type=Path)
    p.add_argument('--actor-pid',required=True,type=int)
    p.add_argument('--baseline',required=True,type=Path)
    p.add_argument('--service-pid',type=int)
    p.add_argument('--service-config',type=Path)
    p.add_argument('--env-gpu',type=int,default=6)
    p.add_argument('--server')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--gpu',required=True,type=int)
    args=p.parse_args()
    if (args.service_pid is None)==(args.service_config is None):raise ValueError('Choose an existing service PID or an owned service config')
    args.output.mkdir(parents=True,exist_ok=False)
    owned_service=None;env_log=None
    def status(stage,**kw):
        record=dict(stage=stage,time=time.time(),**kw)
        (args.output/'status.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    try:
        actor=identity(args.actor_pid)
        if actor is None or 'panovln_dapo.actor' not in actor[1] or str(args.actor_run) not in actor[1]:
            raise ValueError('Actor PID is not the specified live update')
        service=identity(args.service_pid) if args.service_pid is not None else None
        if args.service_pid is not None and (service is None or 'panovln_dapo.env_server' not in service[1]):raise ValueError('Evaluation service identity mismatch')
        status('waiting_for_actor',actor_pid=args.actor_pid)
        while True:
            current=identity(args.actor_pid)
            if current is None:break
            if current!=actor:raise RuntimeError('Actor PID reused')
            time.sleep(10)
        deadline=time.monotonic()+30
        while True:
            stage=json.loads((args.actor_run/'status.json').read_text())['stage']
            if stage=='update_exported':break
            if stage=='failed' or time.monotonic()>deadline:raise RuntimeError('Actor did not finish successful export')
            time.sleep(1)
        from .compare import read_run,compare
        base_config,_=read_run(args.baseline)
        from .checkpoint_audit import audit
        import torch
        torch.set_num_threads(2)
        checkpoint=args.actor_run/'checkpoint'
        status('auditing_checkpoint')
        audit_report=audit(update_initial_checkpoint(args.actor_run),checkpoint)
        (args.output/'checkpoint_audit.json').write_text(json.dumps(audit_report,indent=2)+'\n')
        server=args.server or base_config['server']
        if args.service_config is not None:
            address=urllib.parse.urlparse(server)
            if address.hostname!='127.0.0.1' or address.port is None:raise ValueError('Owned service requires an explicit localhost port')
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
            if any(memory.get(g,100000)>1000 for g in (args.gpu,args.env_gpu)):raise RuntimeError('Evaluation GPUs not idle')
            import socket
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1',address.port))==0:raise RuntimeError('Evaluation port already occupied')
            env_command=[sys.executable,'-u','-m','panovln_dapo.env_server','--config',str(args.service_config),
                         '--gpu-ids',str(args.env_gpu),'--pool-size','1','--port',str(address.port)]
            env_log=(args.output/'environment.log').open('x')
            owned_service=subprocess.Popen(env_command,stdout=env_log,stderr=subprocess.STDOUT,start_new_session=True)
            status('environment_starting',pid=owned_service.pid)
            deadline=time.monotonic()+180
            while True:
                if owned_service.poll() is not None:raise RuntimeError('Owned environment exited during startup')
                try:
                    with urllib.request.urlopen(server+'/health',timeout=2) as response:
                        if response.status==200:break
                except (OSError,TimeoutError):pass
                if time.monotonic()>deadline:raise RuntimeError('Owned environment readiness timeout')
                time.sleep(1)
        elif identity(args.service_pid)!=service:raise RuntimeError('Evaluation service identity changed')
        with urllib.request.urlopen(server+'/stats',timeout=10) as response:stats=json.load(response)
        if stats['active_sessions']!=0:raise RuntimeError('Evaluation service is busy')
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
        if memory.get(args.gpu,100000)>1000:raise RuntimeError('Evaluation GPU is not idle')
        candidate=args.output/'candidate'
        command=[sys.executable,'-u','-m','panovln_dapo.evaluate','--checkpoint',str(checkpoint),
                 '--manifest',base_config['manifest'],'--server',server,'--max-episodes',str(base_config['max_episodes']),
                 '--output',str(candidate)]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        snapshot=args.output/'source_snapshot';snapshot.mkdir();hashes={}
        for source in Path('rl/panovln_dapo').glob('*.py'):
            data=source.read_bytes();(snapshot/source.name).write_bytes(data);hashes[source.name]=hashlib.sha256(data).hexdigest()
        (args.output/'run_manifest.json').write_text(json.dumps(dict(command=command,gpu=args.gpu,source_sha256=hashes,
            environment_config_sha256=hashlib.sha256(args.service_config.read_bytes()).hexdigest() if args.service_config else None,
            baseline_config_sha256=hashlib.sha256((args.baseline/'config.json').read_bytes()).hexdigest()),indent=2)+'\n')
        with (args.output/'evaluation.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            status('evaluation_running',pid=child.pid);code=child.wait()
        if code:raise RuntimeError(f'Evaluation exited with status {code}')
        report=compare(args.baseline,candidate)
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        status('comparison_complete',report=str(args.output/'comparison.json'))
    except BaseException as exc:
        status('failed',error=str(exc));raise
    finally:
        if owned_service is not None and owned_service.poll() is None:
            owned_service.terminate()
            try:owned_service.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(owned_service.pid,signal.SIGKILL);owned_service.wait(timeout=5)
        if env_log is not None:env_log.close()


if __name__=='__main__':main()
