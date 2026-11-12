"""Pano loader adapter around SimpleVLN's session/worker pool.

Start only on authorized GPUs. This module does not start workers on import.
The child target installs the adapter after spawn and before Habitat imports.
"""
import argparse
import os
from .simplevln import add_source_path


def worker_loop(worker_id, config_path, cmd_queue, resp_queue, gpu_id=-1):
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id)
    add_source_path()
    from recipe.vln_simplevln.env_server import worker
    worker._load_env = load_pano_env
    worker.worker_loop(worker_id,config_path,cmd_queue,resp_queue,gpu_id)


def load_pano_env(config_path):
    # Import torch before Habitat, matching Pano's native evaluator.
    import torch
    import habitat
    from habitat import Env
    from habitat.config.default import get_config
    from habitat.config.default_structured_configs import CollisionsMeasurementConfig
    from habitat_extensions import task, measures
    from recipe.vln_simplevln.env_server import worker
    cfg=get_config(config_path)
    if cfg.habitat.dataset.type not in ('R2RVLNCE-v1','RxRVLNCE-v1'):
        raise ValueError('Pano environment requires the native dataset loader')
    if str(cfg.habitat.dataset.data_path).endswith('.parquet'):
        raise ValueError('Restore Parquet source metadata to native JSON before starting Habitat')
    sensor=cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    if sensor.type != 'HabitatSimEquirectangularRGBSensor':
        raise ValueError(f'Expected ERP sensor, got {sensor.type}')
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update({'collisions':CollisionsMeasurementConfig()})
    dataset=habitat.datasets.make_dataset(id_dataset=cfg.habitat.dataset.type,config=cfg.habitat.dataset)
    # Canonical full paths avoid basename collisions in merged scene collections.
    def scene_key(scene): return task.resolve_episode_scene(str(scene),cfg.habitat.dataset.scenes_dir)
    worker._scene_key=scene_key
    full,bare={},{}
    for episode in dataset.episodes:
        key=f'{scene_key(episode.scene_id)}::{episode.episode_id}'
        full.setdefault(key,[]).append(episode)
        bare.setdefault(str(episode.episode_id),[]).append(episode)
    return Env(config=cfg,dataset=dataset),full,bare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--gpu-ids',required=True)
    parser.add_argument('--pool-size',type=int,default=2)
    parser.add_argument('--port',type=int,default=8013)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--session-ttl-sec',type=float,default=1800)
    args=parser.parse_args()
    gpu_ids=[int(x) for x in args.gpu_ids.split(',')]
    if args.pool_size<1 or any(x<0 for x in gpu_ids): raise ValueError('Invalid worker/GPU configuration')
    add_source_path()
    from recipe.vln_simplevln.env_server import server
    import uvicorn
    server.worker_loop=worker_loop
    server.set_config(args.config,pool_size=args.pool_size,session_ttl_sec=args.session_ttl_sec,gpu_ids=gpu_ids)
    uvicorn.run(server.app,host=args.host,port=args.port)


if __name__=='__main__': main()
