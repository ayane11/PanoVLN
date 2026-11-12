from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from panovln_dapo.actor import load_batch
from panovln_dapo.episode import Generation


class ActorInputTests(unittest.TestCase):
    def payload(self,rewards):
        return {'checkpoint':'model','policy_version':0,'cursor':{},'trajectories':[
            {'group_uid':'g','trajectory_uid':str(i),'instruction':'Go','reward':r,'image_buffer':['image'],
             'decisions':[{'env_step_before':0,'generation':asdict(Generation('forward',prompt_ids=[1],response_ids=[2],response_logprobs=[-.1],response_mask=[1],image_indices=[0]))}]}
            for i,r in enumerate(rewards)]}

    def load(self,payload):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'batch.json';path.write_text(json.dumps(payload))
            return load_batch(path)

    def test_real_reward_contrast(self):
        _,rows=self.load(self.payload([0,1]))
        self.assertLess(rows[0]['advantage'],0);self.assertGreater(rows[1]['advantage'],0)

    def test_uniform_groups_not_updated(self):
        with self.assertRaisesRegex(ValueError,'zero'):self.load(self.payload([1,1]))

    def test_empty_batch_not_updated(self):
        with self.assertRaisesRegex(ValueError,'No useful'):self.load(self.payload([]))

class ReplayFrameTests(unittest.TestCase):
    def test_only_selected_frames_are_decoded(self):
        import base64,io
        from PIL import Image
        from panovln_dapo.actor import ReplayFrames
        buffer=io.BytesIO();Image.new('RGB',(8,4)).save(buffer,format='JPEG')
        encoded=base64.b64encode(buffer.getvalue()).decode()
        frames=ReplayFrames([encoded]*100,75)
        self.assertEqual(len(frames),75)
        self.assertIs(frames[-1],frames[74])
        self.assertEqual(len(frames.cache),1)
        self.assertEqual(frames[0].size,(8,4))
        self.assertEqual(len(frames.cache),2)
        with self.assertRaises(IndexError):frames[75]

class GroupIntegrityTests(unittest.TestCase):
    payload = ActorInputTests.payload
    load = ActorInputTests.load
    def test_singleton_is_not_a_contrast_group(self):
        with self.assertRaisesRegex(ValueError,'Incomplete contrast'): self.load(self.payload([1]))

    def test_mixed_episode_group_rejected(self):
        payload=self.payload([0,1])
        payload['trajectories'][1]['instruction']='Different episode'
        with self.assertRaisesRegex(ValueError,'mixes different'): self.load(payload)

    def test_group_metrics_must_match(self):
        payload=self.payload([0,1]);payload['metrics']={'valid_groups':2}
        with self.assertRaisesRegex(ValueError,'group count'): self.load(payload)
