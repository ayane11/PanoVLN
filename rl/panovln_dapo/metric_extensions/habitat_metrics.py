"""Opt-in published-GT Habitat metrics with distinct registry type names."""
from dataclasses import dataclass
import gzip
import json
from pathlib import Path

import torch  # Follow Pano's Habitat import order.
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import Success
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .reference_dtw import reference_ndtw


@registry.register_measure
class SimpleVLNReferenceNDTW(Measure):
    cls_uuid = 'ndtw'

    def __init__(self, *args, sim, config, dataset, **kwargs):
        self._sim, self._config = sim, config
        split = dataset.config.split
        self.gt_path = Path(config.gt_path.format(split=split))
        with gzip.open(self.gt_path, 'rt') as stream:
            self.gt = json.load(stream)
        super().__init__()

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        key = str(episode.episode_id)
        if key not in self.gt:
            raise ValueError('Published reference missing for episode '+key)
        self.reference = self.gt[key]['locations']
        self.positions = []
        self.update_metric()

    def update_metric(self, *args, **kwargs):
        position = self._sim.get_agent_state().position.tolist()
        if not self.positions or position != self.positions[-1]:
            self.positions.append(position)
        self._metric = reference_ndtw(self.positions, self.reference,
                                      success_distance=float(self._config.success_distance))


@registry.register_measure
class SimpleVLNReferenceSDTW(Measure):
    cls_uuid = 'sdtw'

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(self.uuid, [Success.cls_uuid, 'ndtw'])
        self.update_metric(task=task)

    def update_metric(self, *args, task, **kwargs):
        metrics = task.measurements.measures
        self._metric = float(metrics[Success.cls_uuid].get_metric())*float(metrics['ndtw'].get_metric())


@dataclass
class ReferenceNDTWConfig(MeasurementConfig):
    type: str = 'SimpleVLNReferenceNDTW'
    gt_path: str = MISSING
    success_distance: float = 3.


@dataclass
class ReferenceSDTWConfig(MeasurementConfig):
    type: str = 'SimpleVLNReferenceSDTW'


cs = ConfigStore.instance()
cs.store(group='habitat/task/measurements', name='simplevln_reference_ndtw', node=ReferenceNDTWConfig)
cs.store(group='habitat/task/measurements', name='simplevln_reference_sdtw', node=ReferenceSDTWConfig)
