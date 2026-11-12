"""Gate one official-data update and paired evaluation; own only child services."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def validate_adopted_collection(run, root, checkpoint, hashes):
    run,root,checkpoint=Path(run).resolve(),Path(root).resolve(),Path(checkpoint).resolve()
    previous=json.loads((run/'config.json').read_text())
    manifest=root/'data/train.parquet'
    if previous['manifest_sha256']!=hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError('Adopted collection manifest mismatch')
    for name in ('collect.py','backend.py','inputs.py','episode.py','sampling.py','data.py','simplevln.py'):
        if previous['source_sha256'][name]!=hashes[name]:
            raise ValueError('Adopted collection implementation changed: '+name)
    collection=run/'collection'
    cfg=json.loads((collection/'config.json').read_text())
    if Path(cfg['checkpoint']).resolve()!=checkpoint or Path(cfg['manifest']).resolve()!=manifest:
        raise ValueError('Adopted collection policy/data mismatch')
    expected=dict(epochs=20,seed=42,rollout_n=8,target_groups=8,generation_groups=1,
                  max_generation_batches=0,max_response_tokens=64,max_env_steps=500,
                  max_decisions=500,temperature=1.,progress_coef=0.,min_free_gib=70.)
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Adopted collection sampling configuration mismatch')
    return collection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-name', default='experiment_01')
    p.add_argument('--memory-limit-gib', type=float, default=40.)
    p.add_argument('--verified-model-gate',type=Path,help='Reuse a complete gate with matching checkpoint, manifest and input implementation')
    p.add_argument('--adopt-run',type=Path,help='Preserve an existing live collection when changing actor resources')
    p.add_argument('--collector-pid',type=int)
    p.add_argument('--service-pid',type=int)
    p.add_argument('--render-pid', type=int, required=True)
    p.add_argument('--render-audit', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--eval-config', type=Path, required=True)
    p.add_argument('--actor-gpus', default='1,3,5')
    p.add_argument('--rollout-gpu', type=int, default=1)
    p.add_argument('--env-gpu', type=int, default=3)
    p.add_argument('--port', type=int, default=8023)
    args = p.parse_args()
    if bool(args.adopt_run)!=bool(args.collector_pid and args.service_pid):
        raise ValueError('Adoption requires run, collector PID and owned service PID together')
    args.root = args.root.resolve()
    if Path(args.run_name).name != args.run_name or args.run_name in ('.','..'):
        raise ValueError('Run name must be a directory basename')
    out = args.root/args.run_name; out.mkdir(exist_ok=False)
    from .first_update import identity
    from .train_loop import atomic_json
    source_dir = Path(__file__).parent
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    hashes = {}
    for path in source_dir.glob('*.py'):
        content = path.read_bytes(); (snapshot/path.name).write_bytes(content)
        hashes[path.name] = hashlib.sha256(content).hexdigest()
    atomic_json(out/'config.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256=hashes, initial_policy='Original Pano checkpoint; fresh official-data cursor and optimizer',
        rollout_n=8,target_groups=8,epochs=20,updates=1,memory_limit_gib=args.memory_limit_gib,
        manifest_sha256=hashlib.sha256((args.root/'data/train.parquet').read_bytes()).hexdigest()))
    child_service = None
    service_log = None
    def status(stage, **kwargs):
        record = dict(stage=stage,time=time.time(),**kwargs)
        atomic_json(out/'status.json',record);print(json.dumps(record),flush=True)
    def check_sources():
        current={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source_dir.glob('*.py')}
        if current!=hashes: raise RuntimeError('Source changed after experiment snapshot')
    def idle(*gpus):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        used={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines()}
        if any(used.get(g,100000)>1000 for g in gpus): raise RuntimeError(f'Required GPU busy: {used}')
    def run(stage, command, devices):
        check_sources()
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=devices
        with (out/(stage+'.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',*command],env=env,stdout=log,stderr=subprocess.STDOUT)
            status(stage,pid=child.pid,command=command)
            code=child.wait()
        if code: raise RuntimeError(f'{stage} failed ({code}); inspect {stage}.log')
    def stop_service():
        nonlocal child_service,service_log
        if child_service is not None:
            # This new session belongs exclusively to this supervisor. Never
            # look up or terminate services by port, GPU, name or other PID.
            if child_service.poll() is None:
                os.killpg(child_service.pid,signal.SIGTERM)
                try: child_service.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child_service.pid,signal.SIGKILL);child_service.wait(timeout=10)
            child_service=None
        if service_log is not None: service_log.close();service_log=None
    def start_service(config, name):
        nonlocal child_service,service_log
        idle(args.env_gpu)
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',args.port))==0: raise RuntimeError('Dedicated port occupied')
        command=[sys.executable,'-u','-m','panovln_dapo.env_server','--config',str(config),
                 '--gpu-ids',str(args.env_gpu),'--pool-size','1','--port',str(args.port)]
        service_log=(out/(name+'.log')).open('x')
        env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
        child_service=subprocess.Popen(command,env=env,stdout=service_log,stderr=subprocess.STDOUT,start_new_session=True)
        status(name,pid=child_service.pid)
        deadline=time.monotonic()+300
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic()<deadline:
            if child_service.poll() is not None:raise RuntimeError('Owned service exited')
            try:
                with opener.open(f'http://127.0.0.1:{args.port}/health',timeout=2) as response:
                    if response.status==200:return
            except (OSError,TimeoutError): pass
            time.sleep(1)
        raise RuntimeError('Owned service readiness timed out')
    try:
        assets=json.loads((args.root/'assets_audit/complete.json').read_text())
        if not assets['passed'] or assets['episodes']!=6901 or assets['scenes']!=725:
            raise RuntimeError('Full asset gate missing')
        render_identity=identity(args.render_pid)
        if render_identity is not None and 'panovln_dapo.audit_official_render' not in render_identity[1]:
            raise RuntimeError('Render audit PID mismatch')
        status('waiting_for_full_render_audit',render_pid=args.render_pid)
        while identity(args.render_pid) is not None:
            if identity(args.render_pid)!=render_identity:raise RuntimeError('Render PID reused')
            time.sleep(10)
        report=json.loads((args.render_audit/'complete.json').read_text())
        if not report['passed'] or report['scenes']!=725:raise RuntimeError('Full render gate failed')
        if args.verified_model_gate:
            gate=args.verified_model_gate.resolve()
            previous=json.loads((gate.parent/'config.json').read_text())
            if Path(previous['arguments']['checkpoint']).resolve()!=args.checkpoint.resolve():raise RuntimeError('Gate checkpoint mismatch')
            if previous['manifest_sha256']!=hashlib.sha256((args.root/'data/train.parquet').read_bytes()).hexdigest():raise RuntimeError('Gate manifest mismatch')
            for name in ('official_model_gate.py','backend.py','inputs.py','env_server.py','episode.py','simplevln.py'):
                if previous['source_sha256'][name]!=hashes[name]:raise RuntimeError('Gate input implementation changed: '+name)
            report=json.loads((gate/'complete.json').read_text())
            if not report['passed'] or {r['category'] for r in report['reports']}!={'r2r','rxr','mp3d_other','hm3d'}:
                raise RuntimeError('Incomplete reusable model gate')
            if any(not (0<=r['max_error']<=1e-3 and 0<=r['mean_error']<=1e-4) for r in report['reports']):
                raise RuntimeError('Reusable model probability gate failed')
            atomic_json(out/'reused_model_gate.json',dict(source=str(gate),report=report,
                report_sha256=hashlib.sha256((gate/'complete.json').read_bytes()).hexdigest()))
        else:
            idle(args.rollout_gpu,args.env_gpu)
            run('model_gate',['-m','panovln_dapo.official_model_gate','--config',str(args.root/'environment.yaml'),
                '--checkpoint',str(args.checkpoint),'--output',str(out/'model_gate')],f'{args.rollout_gpu},{args.env_gpu}')
            if not json.loads((out/'model_gate/complete.json').read_text())['passed']:raise RuntimeError('Model gate failed')
        # This run intentionally starts a new data distribution from the original
        # checkpoint. Never transplant the old R2R cursor/optimizer implicitly.
        if (args.checkpoint/'vln_progress.json').exists():raise RuntimeError('Expected original initialization, not old data continuation')
        # actor needs 70 GiB BEFORE export (not an additional post-export floor).
        # Start with 10 GiB sampling slack; each write also enforces the 70 GiB
        # reserve including its exact serialized size, even for the final batch.
        if shutil.disk_usage(args.root).free < 80*1024**3:raise RuntimeError('Need 70 GiB export reserve plus 10 GiB initial sampling slack')
        server=f'http://127.0.0.1:{args.port}'
        if args.adopt_run:
            collection=validate_adopted_collection(args.adopt_run,args.root,args.checkpoint,hashes)
            collector_identity=identity(args.collector_pid)
            service_identity=identity(args.service_pid)
            if collector_identity is not None and ('panovln_dapo.collect' not in collector_identity[1] or str(collection) not in collector_identity[1]):
                raise RuntimeError('Adopted collector PID mismatch')
            if service_identity is None or 'panovln_dapo.env_server' not in service_identity[1] or str(args.root/'environment.yaml') not in service_identity[1] or f'--port {args.port}' not in service_identity[1]:
                raise RuntimeError('Adopted service PID mismatch')
            if os.getpgid(args.service_pid)!=args.service_pid:
                raise RuntimeError('Adopted service is not its own process group')
            atomic_json(out/'adopted_collection.json',dict(source=str(collection),collector_pid=args.collector_pid,
                collector_identity=collector_identity,service_pid=args.service_pid,service_identity=service_identity))
            status('waiting_for_existing_collection',collector_pid=args.collector_pid,service_pid=args.service_pid,collection=str(collection))
            while identity(args.collector_pid) is not None:
                if identity(args.collector_pid)!=collector_identity:raise RuntimeError('Collector PID reused')
                time.sleep(10)
            if (collection/'failure.json').exists() or not (collection/'complete.json').exists():
                raise RuntimeError('Adopted collection did not complete successfully')
            if identity(args.service_pid)!=service_identity:raise RuntimeError('Adopted service identity changed')
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(server+'/stats',timeout=10) as response:stats=json.load(response)
            if stats['active_sessions']!=0:raise RuntimeError('Adopted environment still busy')
            # Stop only the explicitly adopted service's verified isolated group.
            os.killpg(args.service_pid,signal.SIGTERM)
            deadline=time.monotonic()+60
            while identity(args.service_pid) is not None and time.monotonic()<deadline:time.sleep(1)
            if identity(args.service_pid) is not None:raise RuntimeError('Adopted environment did not exit')
        else:
            collection=out/'collection'
            idle(args.rollout_gpu)
            start_service(args.root/'environment.yaml','training_environment')
            run('collecting',['-m','panovln_dapo.collect','--checkpoint',str(args.checkpoint),
                '--manifest',str(args.root/'data/train.parquet'),'--server',server,'--output',str(collection),
                '--epochs','20','--rollout-n','8','--target-groups','8','--generation-groups','1',
                '--max-generation-batches','0','--max-response-tokens','64','--min-free-gib','70'],str(args.rollout_gpu))
            stop_service()
        metrics=json.loads((collection/'complete.json').read_text())
        if metrics['valid_groups']!=8 or metrics['partial_batch']:raise RuntimeError('Expected a complete 8-group batch')
        time.sleep(3)
        run('updating',['-m','panovln_dapo.run_actor','--batch',str(collection/'selected_batch.json'),
            '--checkpoint',str(args.checkpoint),'--output',str(out/'actor'),'--gpus',args.actor_gpus,
            '--shared-gpus','--memory-limit-gib',str(args.memory_limit_gib),'--mini-batch-decisions','128'], '')
        run('auditing_export',['-m','panovln_dapo.checkpoint_audit','--initial',str(args.checkpoint),
            '--candidate',str(out/'actor/checkpoint'),'--output',str(out/'checkpoint_audit.json')], '')
        from .compare import read_run, compare
        baseline_config,_=read_run(args.baseline)
        idle(args.rollout_gpu)
        start_service(args.eval_config,'evaluation_environment')
        run('evaluating',['-m','panovln_dapo.evaluate','--checkpoint',str(out/'actor/checkpoint'),
            '--manifest',baseline_config['manifest'],'--server',server,
            '--max-episodes',str(baseline_config['max_episodes']),'--output',str(out/'evaluation')],str(args.rollout_gpu))
        stop_service()
        comparison=compare(args.baseline,out/'evaluation')
        atomic_json(out/'comparison.json',comparison)
        status('comparison_complete',checkpoint=str(out/'actor/checkpoint'),comparison=str(out/'comparison.json'))
    except BaseException as exc:
        status('failed',error=str(exc));raise
    finally:
        stop_service()


if __name__=='__main__':main()
