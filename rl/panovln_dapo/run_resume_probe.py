import argparse,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
parser=argparse.ArgumentParser(description='Supervise a configurable GPU-count restore/training probe; never starts full training.')
parser.add_argument('--gpus',required=True)
parser.add_argument('--memory-limit-gib',required=True,type=float)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--batch',required=True)
parser.add_argument('--steps',type=int,default=2)
args=parser.parse_args();gpus=[int(g) for g in args.gpus.split(',')]
if not gpus or len(set(gpus))!=len(gpus) or min(gpus)<0:raise ValueError('Invalid GPU list')
if not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib<=0 or args.steps<2:raise ValueError('Invalid probe budget')
root=args.output;root.mkdir(parents=True,exist_ok=False)
(root/'supervise.py').write_bytes(Path(__file__).read_bytes())
child=None

def status(stage,**kw):
    payload=dict(stage=stage,time=time.time(),supervisor_pid=os.getpid(),**kw)
    tmp=root/'status.tmp';tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(root/'status.json')
    print(json.dumps(payload),flush=True)

def stop(signum,frame):raise SystemExit(f'Signal {signum}')
signal.signal(signal.SIGTERM,stop)
try:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
    if any(free.get(g,0)<(args.memory_limit_gib+3)*1024 for g in gpus):raise RuntimeError(f'Need allocator budget plus 3 GiB currently free per selected GPU: {free}')
    (root/'existing_gpu_processes.csv').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],text=True))
    snapshot=root/'source_snapshot';snapshot.mkdir();hashes={}
    for p in Path('rl/panovln_dapo').glob('*.py'):
        data=p.read_bytes();(snapshot/p.name).write_bytes(data);hashes[p.name]=hashlib.sha256(data).hexdigest()
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',f'--nproc-per-node={len(gpus)}','-m','panovln_dapo.shared_resume_probe',
             '--checkpoint',args.checkpoint,
             '--batch',args.batch,
             '--output',str(root/'probe'),'--memory-limit-gib',str(args.memory_limit_gib),'--steps',str(args.steps)]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TORCH_DISTRIBUTED_DEBUG='DETAIL')
    (root/'run_manifest.json').write_text(json.dumps(dict(command=command,source_sha256=hashes,gpus=gpus,allocator_limit_gib=args.memory_limit_gib,scope='Configurable GPU-count diagnostic only; existing tasks preserved'),indent=2)+'\n')
    with (root/'probe.log').open('x') as log:child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    status('probe_running',pid=child.pid,gpus=gpus)
    with (root/'gpu_memory.jsonl').open('x') as log:
        while child.poll() is None:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
            log.write(json.dumps(dict(time=time.time(),gpu_memory=raw))+'\n');log.flush();time.sleep(2)
    if child.returncode:raise RuntimeError(f'Probe exited {child.returncode}; inspect probe.log')
    result=json.loads((root/'probe/complete.json').read_text())
    if not all(result[k] for k in ('optimizer_roundtrip_passed','training_steps_passed','headroom_gate_passed')):raise ValueError('Probe gates did not pass')
    status('probe_passed',result=result)
except BaseException as exc:
    status('failed',error=str(exc));raise
finally:
    if child is not None and child.poll() is None:
        child.terminate()
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:child.kill();child.wait()
