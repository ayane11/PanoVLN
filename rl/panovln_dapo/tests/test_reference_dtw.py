"""Numerical parity against the checked-in SimpleVLN measure source."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from fastdtw import fastdtw
from panovln_dapo.metric_extensions.reference_dtw import reference_ndtw, reference_path_metrics


def source_measure_class():
    path=Path(__file__).resolve().parents[4]/'SimpleVLN/eval/vln/habitat_extensions/measures.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NDTW')
    # Extract the actual class to avoid importing/overwriting Habitat registries.
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    namespace={'Measure':object,'registry':SimpleNamespace(register_measure=lambda cls:cls),
               'np':np,'fastdtw':fastdtw,'euclidean_distance':lambda a,b:np.linalg.norm(np.array(a)-np.array(b))}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace['NDTW']


class ReferenceDTWTests(unittest.TestCase):
    def test_actual_source_random_trajectories(self):
        cls=source_measure_class();rng=np.random.default_rng(42)
        for _ in range(25):
            positions=rng.normal(size=(rng.integers(1,50),3)).tolist()
            positions=[p for p in positions for _ in range(2)]
            gt=rng.normal(size=(rng.integers(1,60),3)).tolist()
            radius=float(rng.uniform(.1,5))
            obj=cls.__new__(cls);obj.dtw_func=fastdtw;obj._success_distance=radius
            obj.gt_json={'example':{'locations':gt}}
            state=SimpleNamespace(position=np.array(positions[0]))
            obj._sim=SimpleNamespace(get_agent_state=lambda:state)
            obj.reset_metric(episode=SimpleNamespace(episode_id='example'))
            for position in positions[1:]:
                state.position=np.array(position);obj.update_metric()
            self.assertAlmostEqual(reference_ndtw(positions,gt,success_distance=radius),obj._metric,places=14)

    def test_preserves_gt_sampling_and_success_weight(self):
        positions=[[0,0,0],[0,0,0],[1,0,0]]
        gt=[[0,0,0],[0,0,0],[1,0,0]]
        self.assertEqual(reference_path_metrics(positions,gt,success=1),{'ndtw':1.,'sdtw':1.})
        self.assertEqual(reference_path_metrics(positions,gt,success=0)['sdtw'],0.)
        self.assertLess(reference_ndtw([[9,0,0]],gt),1.)

    def test_invalid_inputs_fail_instead_of_reporting_zero(self):
        for positions in ([],[[1,2]],[[float('nan'),0,0]]):
            with self.assertRaises(ValueError):reference_ndtw(positions,[[0,0,0]])
        with self.assertRaises(ValueError):reference_ndtw([[0,0,0]],[[0,0,0]],success_distance=0)
        with self.assertRaises(ValueError):reference_path_metrics([[0,0,0]],[[0,0,0]],success=.5)
