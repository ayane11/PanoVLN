"""Single model owner for dynamic decision batching across environment sessions."""
import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor


class InferenceQueue:
    def __init__(self, policy, batch_size=1, wait_ms=2.):
        if batch_size<1 or not math.isfinite(wait_ms) or wait_ms<0:
            raise ValueError('Invalid inference batching configuration')
        self.policy,self.batch_size,self.wait_ms=policy,batch_size,wait_ms
        self.queue=asyncio.Queue()
        self.task=None
        self.closed=False
        self.failure=None
        self.executor=None
        self.batch_histogram={}
        self.inference_seconds=0.

    async def __aenter__(self):
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pano-inference')
        self.task=asyncio.create_task(self._run())
        return self

    async def __aexit__(self,*exc):
        self.closed=True
        self.task.cancel()
        await asyncio.gather(self.task,return_exceptions=True)
        self.executor.shutdown(wait=True)
        self._fail_pending(RuntimeError('Inference queue closed'))

    def _fail_pending(self,error):
        while not self.queue.empty():
            _,future=self.queue.get_nowait()
            if not future.done(): future.set_exception(error)

    async def generate(self,instruction,frames):
        if self.failure is not None: raise self.failure
        if self.closed or self.task is None: raise RuntimeError('Inference queue is not running')
        future=asyncio.get_running_loop().create_future()
        await self.queue.put(((instruction,frames),future))
        return await future

    def _generate_sync(self,requests):
        device=getattr(getattr(self.policy,'model',None),'device',None)
        if getattr(device,'type',None)=='cuda':
            import torch
            with torch.cuda.device(device):
                return self.policy.generate_batch(requests)
        return self.policy.generate_batch(requests)

    def metrics(self):
        calls=sum(self.batch_histogram.values())
        decisions=sum(size*count for size,count in self.batch_histogram.items())
        return dict(inference_calls=calls,inference_decisions=decisions,
                    inference_max_batch=max(self.batch_histogram,default=0),
                    inference_mean_batch=decisions/calls if calls else 0.,
                    inference_batch_histogram=dict(self.batch_histogram),
                    inference_wall_seconds=self.inference_seconds)

    async def _run(self):
        batch=[]
        try:
            while True:
                batch=[await self.queue.get()]
                if self.batch_size>1:
                    await asyncio.sleep(self.wait_ms/1000)
                while len(batch)<self.batch_size and not self.queue.empty():
                    batch.append(self.queue.get_nowait())
                batch=[item for item in batch if not item[1].cancelled()]
                if not batch: continue
                # One executor thread owns all forwards; HTTP coroutines keep
                # running while native generation executes. Join before unload.
                started=time.monotonic()
                future=asyncio.get_running_loop().run_in_executor(
                    self.executor,self._generate_sync,[request for request,_ in batch])
                try:
                    values=await asyncio.shield(future)
                except asyncio.CancelledError:
                    try:await future
                    except Exception:pass
                    raise
                self.inference_seconds+=time.monotonic()-started
                self.batch_histogram[len(batch)]=self.batch_histogram.get(len(batch),0)+1
                if len(values)!=len(batch): raise ValueError('Inference batch result count mismatch')
                for (_,future),value in zip(batch,values):
                    if not future.done(): future.set_result(value)
                batch=[]
        except BaseException as error:
            self.failure=error
            for _,future in batch:
                if not future.done(): future.set_exception(error)
            self._fail_pending(error)
            if isinstance(error,asyncio.CancelledError): raise
