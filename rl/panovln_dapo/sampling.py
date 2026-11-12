"""Bounded online sampling and resumable episode order for DAPO.

A sampling round never mixes policy versions. Failed trajectories fail the round;
no silently dropped samples or fabricated failures. Checkpoint only at round/update
boundaries, with the cursor and optimizer saved together.
"""
import asyncio
import traceback
import copy
import math
import random
from dataclasses import dataclass
from .data import digest


class EpisodeCursor:
    def __init__(self, rows, *, seed=42, epochs=1, shuffle=True):
        if not rows or epochs < 1:
            raise ValueError('Need episodes and a positive epoch budget')
        self.rows = rows
        self.seed, self.epochs, self.shuffle = seed, epochs, shuffle
        self.fingerprint = digest(rows)
        self.epoch = self.offset = 0

    def state_dict(self):
        return dict(fingerprint=self.fingerprint, seed=self.seed, epochs=self.epochs,
                    shuffle=self.shuffle, epoch=self.epoch, offset=self.offset)

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ('fingerprint','seed','epochs','shuffle'):
            if state[key] != expected[key]:
                raise ValueError(f'Cursor configuration mismatch: {key}')
        epoch, offset = state['epoch'], state['offset']
        if not isinstance(epoch,int) or not isinstance(offset,int) or not 0 <= epoch <= self.epochs or not 0 <= offset < len(self.rows) or (epoch == self.epochs and offset):
            raise ValueError('Invalid saved cursor')
        self.epoch, self.offset = epoch, offset

    @property
    def exhausted(self): return self.epoch >= self.epochs

    def take(self, count):
        if count < 1: raise ValueError('Batch size must be positive')
        if self.exhausted: return []
        # A generation batch stops at the epoch boundary, as a dataloader does.
        order = list(range(len(self.rows)))
        if self.shuffle: random.Random(self.seed+self.epoch).shuffle(order)
        end = min(self.offset+count, len(order))
        result = [(f'{self.epoch}:{position}', self.rows[order[position]]) for position in range(self.offset,end)]
        self.offset = end
        if end == len(order): self.epoch += 1; self.offset = 0
        return result


@dataclass(frozen=True)
class SamplingConfig:
    rollout_n: int = 4
    target_groups: int = 8
    generation_groups: int = 8
    max_generation_batches: int = 4  # zero: unlimited until data exhaustion
    concurrency: int = 2
    filter_groups: bool = True
    epsilon: float = 1e-8
    scheduler: str = "sliding"

    def __post_init__(self):
        if min(self.rollout_n,self.target_groups,self.generation_groups,self.concurrency) < 1 or self.max_generation_batches < 0:
            raise ValueError('Invalid sampling sizes')
        if self.filter_groups and self.rollout_n < 2:
            raise ValueError('DAPO requires at least two trajectories per group')
        if self.scheduler not in ('sliding', 'window'):
            raise ValueError('Scheduler must be sliding or window')
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError('Invalid filter epsilon')


def informative(group, epsilon):
    # Match SimpleVLN _valid_uids: population std over decision rows.
    rewards = [float(t['reward']) for t in group for _ in t['decisions']]
    if not rewards or not all(math.isfinite(r) for r in rewards):
        raise ValueError('Missing/nonfinite trajectory rewards')
    mean = sum(rewards)/len(rewards)
    return math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards)) > epsilon


async def dispatch_rollouts(jobs, rollout, concurrency, scheduler, *, isolate_failures=False, on_error=None):
    """Bound tasks in input order; optional source sliding episode isolation."""
    if concurrency < 1 or scheduler not in ('sliding', 'window'):
        raise ValueError('Invalid rollout dispatch configuration')
    results = [None] * len(jobs)
    pending = {}
    next_index = 0

    def fill():
        nonlocal next_index
        while next_index < len(jobs) and len(pending) < concurrency:
            index = next_index
            pending[asyncio.create_task(rollout(*jobs[index]))] = index
            next_index += 1

    try:
        fill()
        while pending:
            done, _ = await asyncio.wait(pending, return_when=(
                asyncio.FIRST_COMPLETED if scheduler == 'sliding' else asyncio.ALL_COMPLETED))
            for task in done:
                index=pending.pop(task)
                try:results[index]=task.result()
                except Exception as error:
                    if not isolate_failures:raise
                    if on_error is not None:on_error(index,error)
            fill()
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    return results


async def collect_round(cursor, rollout, config=SamplingConfig(), *, policy_version,on_rollout_error=None):
    """rollout(row, group_uid, trajectory_uid, policy_version) -> trajectory dict."""
    initial_state = copy.deepcopy(cursor.state_dict())
    selected, metrics = [], dict(generation_batches=0, sampled_groups=0, filtered_groups=0, discarded_valid_groups=0,failed_trajectories=0,failure_causes={})
    semaphore = asyncio.Semaphore(config.concurrency)

    async def one(row, group_uid, trajectory_uid):
        async with semaphore:
            trajectory = await rollout(row,group_uid,trajectory_uid,policy_version)
            if trajectory.get('group_uid') != group_uid or trajectory.get('trajectory_uid') != trajectory_uid:
                raise ValueError('Rollout identity mismatch')
            if not trajectory.get('decisions'):
                raise ValueError('Cannot train an empty trajectory')
            if not math.isfinite(float(trajectory['reward'])):
                raise ValueError('Nonfinite trajectory reward')
            return trajectory

    try:
        while not cursor.exhausted:
            episodes = cursor.take(config.generation_groups)
            jobs = [(row, f'v{policy_version}:{occurrence}',
                     f'v{policy_version}:{occurrence}:r{replica}')
                    for occurrence, row in episodes for replica in range(config.rollout_n)]
            def report_error(index,error):
                metrics['failed_trajectories']+=1
                reason=f'{type(error).__name__}: {error}'[:200]
                metrics['failure_causes'][reason]=metrics['failure_causes'].get(reason,0)+1
                record=dict(group_uid=jobs[index][1],trajectory_uid=jobs[index][2],
                    error=repr(error),traceback=''.join(traceback.format_exception(type(error),error,error.__traceback__)))
                if on_rollout_error is not None:on_rollout_error(record)
            trajectories = await dispatch_rollouts(jobs, one, config.concurrency, config.scheduler,
                isolate_failures=config.scheduler=='sliding',on_error=report_error)
            if jobs and not any(trajectory is not None for trajectory in trajectories):
                raise RuntimeError('No usable trajectories: all rollouts failed')
            metrics['generation_batches'] += 1
            metrics['sampled_groups'] += len(episodes)
            for offset in range(0,len(trajectories),config.rollout_n):
                group=[trajectory for trajectory in trajectories[offset:offset+config.rollout_n] if trajectory is not None]
                if not group:
                    metrics['filtered_groups']+=1
                    continue
                if config.filter_groups and not informative(group,config.epsilon):
                    metrics['filtered_groups'] += 1
                else: selected.append(group)
            if not config.filter_groups or len(selected)>=config.target_groups:
                break
            if config.max_generation_batches and metrics['generation_batches']>=config.max_generation_batches:
                break
    except BaseException:
        cursor.load_state_dict(initial_state)
        raise
    if config.filter_groups:
        metrics['discarded_valid_groups']=max(0,len(selected)-config.target_groups)
        selected=selected[:config.target_groups]
    metrics.update(valid_groups=len(selected), data_exhausted=cursor.exhausted,
                   partial_batch=len(selected)<config.target_groups, policy_version=policy_version)
    return [trajectory for group in selected for trajectory in group],metrics
