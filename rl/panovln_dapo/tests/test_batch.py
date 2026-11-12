import copy
import unittest
from panovln_dapo.batch import decision_rows,token_count
from panovln_dapo.episode import Generation


def trajectory(uid,reward,steps):
    decisions=[]
    for step in range(steps):
        gen=Generation('forward',prompt_ids=[1],response_ids=[2,3],response_logprobs=[-.1,-.2],response_mask=[1,1],image_indices=[step])
        decisions.append(dict(generation=gen,env_step_before=step))
    return dict(trajectory_uid=uid,group_uid='g',instruction='Go',reward=reward,
                decisions=decisions,image_buffer=['frame']*(steps+1))


class BatchTests(unittest.TestCase):
    def test_long_trajectory_not_extra_group_sample(self):
        rows=decision_rows([trajectory('win',1,1),trajectory('lose',0,3)])
        self.assertEqual(token_count(rows),8)
        self.assertAlmostEqual(rows[0]['advantage'],-3*rows[1]['advantage'],places=5)
        self.assertIs(rows[1]['image_buffer'],rows[2]['image_buffer'])
        self.assertEqual(rows[-1]['frame_count'],3)

    def test_future_frame_rejected(self):
        t=trajectory('x',1,1);t['decisions'][0]['generation'].image_indices=[1]
        with self.assertRaisesRegex(ValueError,'Current'): decision_rows([t])

    def test_missing_logprob_rejected(self):
        t=trajectory('x',1,1);t['decisions'][0]['generation'].response_logprobs=[]
        with self.assertRaisesRegex(ValueError,'misaligned'):decision_rows([t])
