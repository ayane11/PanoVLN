"""Flatten trajectories without dropping replay geometry or token provenance."""
from collections import Counter
import math
import torch
from .algorithms import trajectory_grpo_advantage


def decision_rows(trajectories):
    rows=[]
    seen=set()
    for trajectory in trajectories:
        uid=trajectory['trajectory_uid']
        if uid in seen: raise ValueError('Duplicate trajectory identity')
        seen.add(uid)
        decisions=trajectory['decisions']
        if not decisions: raise ValueError('Empty trajectory')
        if not math.isfinite(trajectory['reward']): raise ValueError('Nonfinite reward')
        for decision in decisions:
            gen=decision['generation']
            n=len(gen.response_ids)
            if not n or len(gen.response_logprobs)!=n or len(gen.response_mask)!=n:
                raise ValueError('Missing or misaligned response data')
            if not gen.prompt_ids or not all(math.isfinite(p) for p in gen.response_logprobs):
                raise ValueError('Invalid prompt/probabilities')
            if any(v not in (0,1) for v in gen.response_mask) or not any(gen.response_mask):
                raise ValueError('Empty or nonbinary response mask')
            before=decision['env_step_before']
            if before >= len(trajectory['image_buffer']) or not gen.image_indices or gen.image_indices[-1]!=before:
                raise ValueError('Current image does not match decision step')
            if any(i<0 or i>before for i in gen.image_indices):
                raise ValueError('Future or invalid replay image')
            rows.append(dict(group_uid=trajectory['group_uid'],trajectory_uid=uid,
                             instruction=trajectory['instruction'],generation=gen,
                             # Shared reference avoids N copies of the trajectory history.
                             image_buffer=trajectory['image_buffer'],frame_count=before+1,
                             reward=float(trajectory['reward']),decision_loss_weight=1/len(decisions)))
    if not rows: raise ValueError('No decisions to optimize')
    # Use one column to compute scalar per-decision advantages; token masking is
    # applied to the actual response when constructing each microbatch.
    scores=torch.tensor([[row['reward']] for row in rows],dtype=torch.float32)
    advantages,_=trajectory_grpo_advantage(scores,torch.ones_like(scores),
        [r['group_uid'] for r in rows],[r['trajectory_uid'] for r in rows],
        [r['decision_loss_weight'] for r in rows])
    for row,adv in zip(rows,advantages[:,0].tolist()): row['advantage']=adv
    return rows


def token_count(rows):
    return sum(sum(row['generation'].response_mask) for row in rows)
