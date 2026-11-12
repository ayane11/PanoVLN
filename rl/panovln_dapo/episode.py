"""Full-episode collection with the SimpleVLN async environment interface.

One decision stores the entire generated response, even when only a prefix is
executed. Frame indices are absolute into image_buffer; the input adapter applies
the Pano memory window. No simulator failures are converted into zero rewards.
"""
from dataclasses import dataclass, field
import math
from .action import action_queue


@dataclass
class Generation:
    action_text: str
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    image_indices: list[int] = field(default_factory=list)
    uncertainty_actions: list[int] | None = None
    uncertainties: list[float] | None = None
    replay_metadata: dict = field(default_factory=dict)


async def run_episode(env, extra_info, decide, *, group_uid, trajectory_uid,
                      max_decisions=500, max_env_steps=500, progress_coef=0.0,
                      actions_per_replan=4, stop_commit_max_actions=10,
                      uncertainty_budget=1.2, replan_action_range=(4, 8)):
    """Own and close env; decide(instruction, all_frames) returns Generation."""
    if max_decisions < 1 or max_env_steps < 1 or not math.isfinite(progress_coef):
        raise ValueError("Invalid trajectory budget/reward configuration")
    frames, decisions = [], []
    steps = 0
    reason = None
    try:
        frames.append(await env.reset(extra_info))
        done = env.done
        while not done and steps < max_env_steps and len(decisions) < max_decisions:
            gen = await decide(env.instruction, list(frames))
            parsed, queue, selection = action_queue(
                gen.action_text, actions_per_replan=actions_per_replan,
                stop_commit_max_actions=stop_commit_max_actions,
                uncertainty_actions=gen.uncertainty_actions, uncertainties=gen.uncertainties,
                uncertainty_budget=uncertainty_budget, replan_action_range=replan_action_range)
            before = steps
            executed = []
            for action in queue[:max_env_steps-steps]:
                response = await env.step([action])
                if response.get("done_reason") in ("worker_error", "timeout"):
                    raise RuntimeError(f"Environment failure: {response['done_reason']}")
                if response.get("executed_actions") != [action]:
                    raise RuntimeError("Environment did not acknowledge the requested atomic action")
                executed.append(action)
                steps += 1
                frames.append(env.current_jpeg_b64())
                done = response["done"]
                reason = response.get("done_reason")
                if action == 0 and not done:
                    raise RuntimeError("Environment failed to terminate after STOP")
                if done:
                    break
            decisions.append(dict(turn_id=len(decisions), generation=gen,
                                  parsed_actions=parsed, planned_actions=queue,
                                  executed_actions=executed, env_step_before=before,
                                  env_step_after=steps, **selection))
        if not done:
            reason = "max_env_steps" if steps >= max_env_steps else "max_decisions"
        # Habitat reports done=True for its own step limit as well as STOP.
        # A server-side time limit is still a truncation, even when the local
        # loop observes a normal environment termination acknowledgement.
        truncated=(not done or reason in ('max_env_steps','max_decisions','max_episode_steps')
                   or (steps >= max_env_steps and reason != 'stop_action'))
        metrics = dict(env.metrics())
        if "success" not in metrics:
            raise ValueError("Missing terminal success metric")
        reward = float(metrics["success"])
        if progress_coef:
            reward += progress_coef * float(metrics["oracle_success"])
        if not math.isfinite(reward):
            raise ValueError("Non-finite terminal reward")
        metrics.update(num_decisions=len(decisions), env_steps=steps)
        return dict(group_uid=group_uid, trajectory_uid=trajectory_uid,
                    scene_id=extra_info["scene_id"], episode_id=str(extra_info["episode_id"]),
                    instruction=env.instruction, reward=reward, metrics=metrics,
                    decisions=decisions, image_buffer=frames, done_reason=reason,
                    truncated=truncated)
    finally:
        await env.close()
