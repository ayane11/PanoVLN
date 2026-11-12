import asyncio
import unittest
import threading
from panovln_dapo.inference_queue import InferenceQueue


class InferenceQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_batches_preserve_request_mapping_and_limit(self):
        class Policy:
            batches=[]
            def generate_batch(self,requests):
                self.batches.append(requests)
                return [instruction for instruction,frames in requests]
        policy=Policy()
        async with InferenceQueue(policy,3,0) as queue:
            values=await asyncio.gather(*(queue.generate(i,[]) for i in range(8)))
        self.assertEqual(values,list(range(8)))
        self.assertEqual([len(batch) for batch in policy.batches],[3,3,2])

    async def test_failure_reaches_all_waiters_and_future_requests(self):
        class Policy:
            def generate_batch(self,requests): raise ValueError('model failed')
        async with InferenceQueue(Policy(),2,0) as queue:
            values=await asyncio.gather(*(queue.generate(i,[]) for i in range(5)),return_exceptions=True)
            self.assertTrue(all(isinstance(value,ValueError) for value in values))
            with self.assertRaisesRegex(ValueError,'model failed'):await queue.generate(6,[])

    async def test_cancelled_request_is_not_dispatched(self):
        class Policy:
            batches=[]
            def generate_batch(self,requests):
                self.batches.extend(requests)
                return [item[0] for item in requests]
        policy=Policy()
        async with InferenceQueue(policy,2,0) as queue:
            cancelled=asyncio.create_task(queue.generate('cancelled',[]))
            await asyncio.sleep(0)
            cancelled.cancel()
            await asyncio.gather(cancelled,return_exceptions=True)
            self.assertEqual(await queue.generate('real',[]),'real')
        self.assertEqual(policy.batches,[('real',[])])


    async def test_model_forward_does_not_block_event_loop(self):
        started=threading.Event();release=threading.Event();finished=threading.Event()
        class Policy:
            def generate_batch(self,requests):
                started.set();release.wait(2);finished.set()
                return [item[0] for item in requests]
        async with InferenceQueue(Policy(),2,0) as queue:
            request=asyncio.create_task(queue.generate('ok',[]))
            self.assertTrue(await asyncio.to_thread(started.wait,1))
            try:
                await asyncio.sleep(.01)
                self.assertFalse(finished.is_set())
            finally:release.set()
            self.assertEqual(await request,'ok')
            self.assertEqual(queue.metrics()['inference_decisions'],1)

    async def test_exit_waits_for_forward_before_model_can_be_unloaded(self):
        started=threading.Event();release=threading.Event();finished=threading.Event()
        class Policy:
            def generate_batch(self,requests):
                started.set();release.wait(2);finished.set()
                return [item[0] for item in requests]
        queue=InferenceQueue(Policy(),1,0);await queue.__aenter__()
        request=asyncio.create_task(queue.generate('ok',[]))
        self.assertTrue(await asyncio.to_thread(started.wait,1))
        closing=asyncio.create_task(queue.__aexit__(None,None,None))
        try:
            await asyncio.sleep(.01)
            self.assertFalse(closing.done())
        finally:release.set()
        await closing
        await asyncio.gather(request,return_exceptions=True)
        self.assertTrue(finished.is_set())
