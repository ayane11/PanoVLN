# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""SimpleVLN trajectory GRPO and verl token-mean dual-clip PPO semantics.

Trajectory function is ported from recipe/vln_simplevln/patches.py.
Loss follows verl/trainer/ppo/core_algos.py compute_policy_loss_vanilla.
Return numerator/count for correct global normalization across microbatches.
"""
from collections import defaultdict
import torch

def trajectory_grpo_advantage(token_level_rewards, response_mask, index, traj_index,
                         decision_weight=None, epsilon=1e-6, norm_adv_by_std_in_grpo=True):
    scores = token_level_rewards.sum(dim=-1)  # (bs,) per-row outcome reward
    bsz = scores.shape[0]

    # group(uid) -> traj_uid -> list of that trajectory's row scores
    group_traj = defaultdict(lambda: defaultdict(list))
    for i in range(bsz):
        group_traj[index[i]][traj_index[i]].append(scores[i])

    group_mean, group_std = {}, {}
    for g, trajs in group_traj.items():
        # one reward per trajectory (rows of a traj are equal when KL is off;
        # mean is the robust aggregation otherwise)
        traj_rewards = torch.stack([torch.stack(v).mean() for v in trajs.values()])
        if traj_rewards.numel() == 1:
            # singleton group: no contrast — match stock GRPO (mean 0, std 1)
            group_mean[g] = scores.new_tensor(0.0)
            group_std[g] = scores.new_tensor(1.0)
        else:
            group_mean[g] = traj_rewards.mean()
            group_std[g] = traj_rewards.std()

    out = scores.clone()
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            out[i] = (scores[i] - group_mean[index[i]]) / (group_std[index[i]] + epsilon)
        else:
            out[i] = scores[i] - group_mean[index[i]]

    # Weight by inverse decision count, then normalize weights to mean one over
    # real rows. Token-mean loss still weights longer responses more heavily.
    if decision_weight is not None:
        w = torch.as_tensor(decision_weight, dtype=out.dtype, device=out.device)
        real = w > 0
        if real.any():
            w = w / w[real].mean()
        out = out * w

    adv = out.unsqueeze(-1) * response_mask
    return adv, adv


def policy_loss_sum(old_log_prob, log_prob, advantages, response_mask,
                    clip_low=0.2, clip_high=0.28, dual_clip=10.0,
                    rollout_is_weights=None):
    if dual_clip <= 1 or not 0 <= clip_low < 1 or clip_high < 0:
        raise ValueError("Invalid PPO clipping configuration")
    if not (old_log_prob.shape == log_prob.shape == advantages.shape == response_mask.shape):
        raise ValueError("PPO tensors must have identical shapes")
    delta = (log_prob - old_log_prob).clamp(-20, 20)
    ratio = delta.exp()
    raw = -advantages * ratio
    clipped = -advantages * ratio.clamp(1-clip_low, 1+clip_high)
    upper = torch.maximum(raw, clipped)
    lower = torch.minimum(-advantages * dual_clip, upper)
    losses = torch.where(advantages < 0, lower, upper)
    if rollout_is_weights is not None:
        losses = losses * rollout_is_weights
    return (losses * response_mask).sum(), response_mask.sum()
