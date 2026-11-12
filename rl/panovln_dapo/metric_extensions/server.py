"""Separate opt-in evaluation service; existing rollout services are untouched."""
import math
import os

from .. import env_server
from ..simplevln import add_source_path

_base_worker_loop = env_server.worker_loop


def extend_metrics(base, info):
    result = dict(base)
    for name in ('ndtw', 'sdtw'):
        if name not in info:
            raise ValueError('Reference evaluation service requires '+name)
        value = float(info[name])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid '+name)
        result[name] = value
    if abs(result['sdtw']-result['success']*result['ndtw']) > 1e-7:
        raise ValueError('sDTW inconsistent with success and nDTW')
    return result


def worker_loop(worker_id, config_path, cmd_queue, resp_queue, gpu_id=-1):
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    from . import habitat_metrics  # Registers in each spawned worker, not only parent.
    add_source_path()
    from recipe.vln_simplevln.env_server import worker
    original = worker._make_metrics
    worker._make_metrics = lambda info, collisions: extend_metrics(original(info, collisions), info)
    _base_worker_loop(worker_id, config_path, cmd_queue, resp_queue, gpu_id)


def install_response_schemas():
    """Extend Pydantic models before FastAPI routes capture their response types."""
    add_source_path()
    from recipe.vln_simplevln.env_server import schemas

    class ReferenceMetrics(schemas.Metrics):
        ndtw: float
        sdtw: float

    class ReferenceCreateSessionResponse(schemas.CreateSessionResponse):
        metrics: ReferenceMetrics

    class ReferenceStepResponse(schemas.StepResponse):
        metrics: ReferenceMetrics

    class ReferenceMetricsResponse(schemas.MetricsResponse):
        metrics: ReferenceMetrics

    schemas.Metrics = ReferenceMetrics
    schemas.CreateSessionResponse = ReferenceCreateSessionResponse
    schemas.StepResponse = ReferenceStepResponse
    schemas.MetricsResponse = ReferenceMetricsResponse


def main():
    install_response_schemas()
    env_server.worker_loop = worker_loop
    env_server.main()


if __name__ == '__main__':
    main()
