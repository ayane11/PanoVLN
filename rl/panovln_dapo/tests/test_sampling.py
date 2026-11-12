import asyncio
import unittest
from panovln_dapo.sampling import EpisodeCursor,SamplingConfig,collect_round


class CursorTests(unittest.TestCase):
    def test_restore_and_epoch_boundaries(self):
        cursor=EpisodeCursor(list(range(5)),epochs=2)
        cursor.take(3); state=cursor.state_dict()
        restored=EpisodeCursor(list(range(5)),epochs=2); restored.load_state_dict(state)
        self.assertEqual(cursor.take(9),restored.take(9))
        self.assertEqual(cursor.take(9),restored.take(9))
        self.assertTrue(cursor.exhausted)
        with self.assertRaises(ValueError): EpisodeCursor(list(range(6)),epochs=2).load_state_dict(state)


class SamplingTests(unittest.IsolatedAsyncioTestCase):
    async def rollout(self,row,group,trajectory,version):
        return dict(group_uid=group,trajectory_uid=trajectory,reward=float(row and trajectory.endswith('r1')),
                    decisions=[{},{}] if trajectory.endswith('r0') else [{}])

    async def test_filter_and_refill(self):
        cursor=EpisodeCursor([0,1,1],shuffle=False)
        result,m=await collect_round(cursor,self.rollout,SamplingConfig(rollout_n=2,target_groups=2,generation_groups=1),policy_version=3)
        self.assertEqual(len(result),4);self.assertEqual(m['filtered_groups'],1)
        self.assertEqual(m['generation_batches'],3);self.assertTrue(m['data_exhausted'])

    async def test_all_zero_capped_without_update(self):
        cursor=EpisodeCursor([0]*10,shuffle=False)
        result,m=await collect_round(cursor,self.rollout,SamplingConfig(rollout_n=2,generation_groups=1,max_generation_batches=2),policy_version=0)
        self.assertEqual(result,[]);self.assertEqual(cursor.offset,2)
        self.assertEqual(m['generation_batches'],2)

    async def test_epoch_end_partial(self):
        cursor=EpisodeCursor([1],shuffle=False)
        result,m=await collect_round(cursor,self.rollout,SamplingConfig(rollout_n=2,target_groups=8),policy_version=0)
        self.assertEqual(len(result),2);self.assertTrue(m['partial_batch'])

    async def test_failure_rolls_back_cursor_and_cancels(self):
        cursor=EpisodeCursor([1,1]);state=cursor.state_dict();closed=[]
        async def fail(row,group,trajectory,version):
            try:
                raise RuntimeError('worker error')
            finally:closed.append(trajectory)
        with self.assertRaises(RuntimeError):
            await collect_round(cursor,fail,SamplingConfig(rollout_n=2),policy_version=0)
        self.assertEqual(cursor.state_dict(),state)
        self.assertTrue(closed)

    async def test_concurrency_is_bounded(self):
        active=peak=0
        async def track(*args):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            try:
                await asyncio.sleep(.001)
                return await self.rollout(*args)
            finally: active-=1
        await collect_round(EpisodeCursor([1]*4),track,SamplingConfig(rollout_n=2,concurrency=2),policy_version=0)
        self.assertEqual(peak,2);self.assertEqual(active,0)

class DispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_sliding_refills_before_slow_peer_finishes(self):
        from panovln_dapo.sampling import dispatch_rollouts
        release=asyncio.Event(); third=asyncio.Event()
        async def run(index):
            if index==0: await release.wait()
            if index==2: third.set()
            return index
        task=asyncio.create_task(dispatch_rollouts([(i,) for i in range(5)],run,2,'sliding'))
        try:
            await asyncio.wait_for(third.wait(),1)
            self.assertFalse(task.done())
        finally:
            release.set()
        self.assertEqual(await task,list(range(5)))

    async def test_window_waits_for_slow_peer(self):
        from panovln_dapo.sampling import dispatch_rollouts
        release=asyncio.Event(); second=asyncio.Event(); started=[]
        async def run(index):
            started.append(index)
            if index==0: await release.wait()
            if index==1: second.set()
            return index
        task=asyncio.create_task(dispatch_rollouts([(i,) for i in range(5)],run,2,'window'))
        try:
            await asyncio.wait_for(second.wait(),1)
            await asyncio.sleep(0)
            self.assertEqual(started,[0,1])
        finally:
            release.set()
        self.assertEqual(await task,list(range(5)))

    async def test_window_failure_cancels_and_rolls_back(self):
        for scheduler in ('window',):
            cursor=EpisodeCursor([1]*4); before=cursor.state_dict(); closed=[]
            async def fail(row,group,trajectory,version):
                try:
                    if trajectory.endswith('r0'): raise RuntimeError('failure')
                    await asyncio.sleep(.01)
                    return await SamplingTests.rollout(self,row,group,trajectory,version)
                finally: closed.append(trajectory)
            with self.assertRaises(RuntimeError):
                await collect_round(cursor,fail,SamplingConfig(rollout_n=2,concurrency=2,scheduler=scheduler),policy_version=0)
            self.assertEqual(before,cursor.state_dict())
            self.assertEqual(len(closed),2)


class SourceFailureIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_sliding_drops_failed_episode_preserves_group_and_cursor(self):
        cursor=EpisodeCursor([1],shuffle=False);failures=[]
        async def rollout(row,group,trajectory,version):
            if trajectory.endswith('r0'):raise RuntimeError('lost reply')
            return dict(group_uid=group,trajectory_uid=trajectory,reward=float(trajectory.endswith('r1')),decisions=[{}])
        trajectories,metrics=await collect_round(cursor,rollout,
            SamplingConfig(rollout_n=3,target_groups=1,generation_groups=1),policy_version=0,
            on_rollout_error=failures.append)
        self.assertEqual(len(trajectories),2)
        self.assertEqual(metrics['failed_trajectories'],1)
        self.assertEqual(metrics['valid_groups'],1)
        self.assertTrue(cursor.exhausted)
        self.assertIn('lost reply',failures[0]['traceback'])

    async def test_all_failed_raises_and_restores_cursor(self):
        cursor=EpisodeCursor([1],shuffle=False);before=cursor.state_dict()
        async def fail(*args):raise RuntimeError('unreachable')
        with self.assertRaisesRegex(RuntimeError,'all rollouts failed'):
            await collect_round(cursor,fail,SamplingConfig(rollout_n=2,generation_groups=1),policy_version=0)
        self.assertEqual(cursor.state_dict(),before)
