import copy
import json
import unittest

from panovln_dapo.official_data import convert_rows
from panovln_dapo.data import restore_annotation


class OfficialDataTests(unittest.TestCase):
    def row(self):
        ep = dict(episode_id='7', trajectory_id=2, scene_id='hm3d/train/00001-test/test.basis.glb',
                  start_position=[0., 0., 0.], start_rotation=[0., 0., 0., 1.],
                  goals=[dict(position=[1., 0., 0.], radius=3.)], reference_path=[[0., 0., 0.], [1., 0., 0.]],
                  instruction=dict(instruction_text='Go forward. '), info=dict(num_turns=1))
        return dict(episode=json.dumps(ep), prompt=[dict(role='user', content='placeholder')],
                    extra_info=dict(episode_id='7', scene_id=ep['scene_id'], instruction='Go forward.'))

    def test_lossless_mixed_scenes_and_shared_ids(self):
        sources = [self.row(), self.row()]
        ep = json.loads(sources[1]['episode'])
        ep['scene_id'] = 'mp3d/test/test.glb'
        ep['instruction']['instruction_tokens'] = [1, 2, 0]
        sources[1]['episode'] = json.dumps(ep)
        sources[1]['extra_info']['scene_id'] = ep['scene_id']
        rows, payload = convert_rows(sources, 'sourcehash')
        self.assertEqual(payload['episodes'], [json.loads(s['episode']) for s in sources])
        self.assertEqual(restore_annotation(rows), payload)
        self.assertNotEqual(rows[0]['extra_info']['episode_key'], rows[1]['extra_info']['episode_key'])

    def test_routing_mismatch_and_duplicates_rejected(self):
        row = self.row()
        row['extra_info']['instruction'] = 'Other instruction'
        with self.assertRaisesRegex(ValueError, 'routing'):
            convert_rows([row], 'sha')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            convert_rows([self.row(), self.row()], 'sha')

    def test_unsafe_scene_rejected(self):
        row = self.row()
        ep = json.loads(row['episode']); ep['scene_id'] = '../test.glb'
        row['episode'] = json.dumps(ep); row['extra_info']['scene_id'] = ep['scene_id']
        with self.assertRaisesRegex(ValueError, 'portable'):
            convert_rows([row], 'sha')
