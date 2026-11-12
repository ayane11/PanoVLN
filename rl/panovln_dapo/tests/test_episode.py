import unittest
from panovln_dapo.action import action_queue, parse_action_sequence
from panovln_dapo.episode import Generation, run_episode


class FakeEnv:
    def __init__(self, done_after=100, fail=False):
        self.instruction = 'Go.'
        self.done = False
        self.actions = []
        self.closed = False
        self.done_after = done_after
        self.fail = fail

    async def reset(self, extra): return 'frame0'
    async def step(self, actions):
        if self.fail: raise TimeoutError('lost response')
        self.actions.extend(actions)
        self.done = actions == [0] or len(self.actions) >= self.done_after
        return {'done': self.done, 'executed_actions': actions,
                'done_reason': 'stop_action' if actions == [0] else ('episode_over' if self.done else None)}
    def current_jpeg_b64(self): return f'frame{len(self.actions)}'
    def metrics(self): return {'success': float(self.actions[-1:] == [0]), 'oracle_success': 1.0}
    async def close(self): self.closed = True


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, env, text, **kwargs):
        self.histories = []
        async def decide(instruction, frames):
            self.histories.append(frames)
            return Generation(text, response_ids=[10,11,12])
        return await run_episode(env, {'scene_id':'scene','episode_id':'1'}, decide,
                                 group_uid='g', trajectory_uid='t', **kwargs)

    async def test_stop_commit_retains_whole_generation(self):
        env = FakeEnv()
        result = await self.collect(env, 'forward left stop right', actions_per_replan=1)
        self.assertEqual(env.actions, [1,2,0])
        self.assertEqual(result['decisions'][0]['generation'].response_ids, [10,11,12])
        self.assertEqual(len(result['image_buffer']), 4)
        self.assertTrue(env.closed)
        self.assertFalse(result['truncated'])

    async def test_actual_execution_not_planned_chunk(self):
        env = FakeEnv(done_after=1)
        result = await self.collect(env, 'forward left right forward')
        self.assertEqual(result['decisions'][0]['executed_actions'], [1])
        self.assertEqual(result['decisions'][0]['planned_actions'], [1,2,3,1])

    async def test_budget_and_history(self):
        env = FakeEnv()
        result = await self.collect(env, 'forward forward', max_env_steps=3)
        self.assertEqual(env.actions, [1,1,1])
        self.assertEqual(self.histories, [['frame0'], ['frame0','frame1','frame2']])
        self.assertTrue(result['truncated'])
        self.assertEqual(result['done_reason'], 'max_env_steps')

    async def test_environment_done_at_step_budget_is_truncated(self):
        result=await self.collect(FakeEnv(done_after=3),'forward',max_env_steps=3)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['done_reason'],'episode_over')
        self.assertEqual(result['metrics']['env_steps'],3)

    async def test_stop_at_step_budget_is_not_truncated(self):
        result=await self.collect(FakeEnv(done_after=3),'forward forward stop',max_env_steps=3)
        self.assertFalse(result['truncated'])
        self.assertEqual(result['done_reason'],'stop_action')

    async def test_empty_output_stops(self):
        env = FakeEnv()
        result = await self.collect(env, 'unparseable')
        self.assertEqual(env.actions, [0])
        self.assertTrue(result['decisions'][0]['invalid_output'])

    async def test_timeout_is_error_not_reward(self):
        env = FakeEnv(fail=True)
        with self.assertRaises(TimeoutError): await self.collect(env, 'forward')
        self.assertTrue(env.closed)
        self.assertEqual(env.actions, [])

    async def test_decision_limit(self):
        result = await self.collect(FakeEnv(), 'forward', max_decisions=2)
        self.assertEqual(result['done_reason'], 'max_decisions')
        self.assertEqual(result['metrics']['env_steps'], 2)


class ActionTests(unittest.TestCase):
    def test_uncertainty_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disagree'):
            action_queue('left forward', actions_per_replan='uncertainty',
                         uncertainty_actions=[1,2], uncertainties=[.1,.2])

    def test_uncertainty_horizon(self):
        parsed, queue, meta = action_queue(' '.join(['forward']*18), actions_per_replan='uncertainty',
                                          uncertainty_actions=[1]*8, uncertainties=[.3]*8)
        self.assertEqual(queue, [1]*4)
        self.assertFalse(meta['stop_committed'])

    def test_parser_aliases_and_limit(self):
        self.assertEqual(parse_action_sequence('<think>right</think>move_forward turn-left stop'), [1,2,0])
        self.assertEqual(len(parse_action_sequence('forward '*30)),18)
