import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from omegaconf import OmegaConf
from panovln_dapo.metric_extensions.server import extend_metrics
from unittest.mock import patch


class MetricServiceTests(unittest.TestCase):
    def test_response_models_retain_metric_fields(self):
        from panovln_dapo.metric_extensions.server import install_response_schemas
        from panovln_dapo.simplevln import add_source_path
        add_source_path()
        from recipe.vln_simplevln.env_server import schemas
        names=('Metrics','CreateSessionResponse','StepResponse','MetricsResponse')
        originals={name:getattr(schemas,name) for name in names}
        try:
            install_response_schemas()
            values=dict(distance_to_goal=1.,success=1.,spl=.9,oracle_success=1.,oracle_navigation_error=.5,collisions=0.,ndtw=.8,sdtw=.8)
            metrics=schemas.Metrics(**values)
            response=schemas.MetricsResponse(session_id='probe',metrics=metrics,done=True)
            self.assertEqual(response.model_dump()['metrics']['ndtw'],.8)
            self.assertEqual(response.model_dump()['metrics']['sdtw'],.8)
            for name in names[1:]:
                self.assertIs(getattr(schemas,name).model_fields['metrics'].annotation,schemas.Metrics)
        finally:
            for name,cls in originals.items():setattr(schemas,name,cls)

    def test_transport_preserves_existing_metrics_and_requires_valid_dtw(self):
        base={'success':1.,'spl':.4,'distance_to_goal':.5,'collisions':2.}
        result=extend_metrics(base,{'ndtw':.8,'sdtw':.8})
        self.assertEqual({k:result[k] for k in base},base)
        self.assertEqual(result['ndtw'],.8)
        self.assertNotIn('ndtw',base)
        for info in ({},{'ndtw':float('nan'),'sdtw':0}, {'ndtw':.8,'sdtw':.2}):
            with self.assertRaises(ValueError):extend_metrics(base,info)

    def test_native_measure_loads_published_reference_and_resets(self):
        from panovln_dapo.metric_extensions.habitat_metrics import SimpleVLNReferenceNDTW
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gt.json.gz'
            with gzip.open(path,'wt') as stream:
                json.dump({'a':{'locations':[[0,0,0],[1,0,0]]},'b':{'locations':[[2,0,0]]}},stream)
            state=SimpleNamespace(position=np.array([0.,0.,0.]))
            metric=SimpleVLNReferenceNDTW(sim=SimpleNamespace(get_agent_state=lambda:state),
                config=OmegaConf.create({'gt_path':str(path),'success_distance':3}),
                dataset=SimpleNamespace(config=SimpleNamespace(split='val_unseen')))
            metric.reset_metric(episode=SimpleNamespace(episode_id='a'))
            state.position=np.array([1.,0.,0.]);metric.update_metric()
            self.assertEqual(metric.get_metric(),1.)
            metric.update_metric();self.assertEqual(len(metric.positions),2)
            state.position=np.array([2.,0.,0.]);metric.reset_metric(episode=SimpleNamespace(episode_id='b'))
            self.assertEqual(metric.get_metric(),1.);self.assertEqual(len(metric.positions),1)
            with self.assertRaisesRegex(ValueError,'missing'):metric.reset_metric(episode=SimpleNamespace(episode_id='absent'))
