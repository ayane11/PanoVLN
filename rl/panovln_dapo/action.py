"""Pano action parsing and queue selection without Habitat/model imports.

Parser and prefix selection mirror src/eval/eval.py; parity tests guard drift.
"""
import re
from typing import Sequence, List
from src.vln_config import VLN_ACTION_SEQUENCE_LENGTH
from src.eval.action_policy import (
    ATOMIC_ACTION_NAMES, STOP_ACTION_ID, validate_action_count,
    select_stop_commit_horizon, select_uncertainty_horizon,
    validate_replan_action_range,
)

ATOMIC_ACTION_VARIANTS = {
    "stop": ("stop",),
    "forward": ("forward", "move_forward", "move forward", "move-forward"),
    "left": ("left", "turn_left", "turn left", "turn-left"),
    "right": ("right", "turn_right", "turn right", "turn-right"),
}

ATOMIC_ACTION_PATTERNS = [
    (
        action_id,
        re.compile(
            r"(?:"
            + "|".join(
                r"(?<![0-9a-z_])" + re.escape(variant.lower()) + r"(?![0-9a-z_])"
                for variant in ATOMIC_ACTION_VARIANTS[action_name]
            )
            + r")"
        ),
    )
    for action_id, action_name in enumerate(ATOMIC_ACTION_NAMES)
]

def select_actions_for_replan(
    action_ids: Sequence[int],
    actions_per_replan: int,
) -> List[int]:
    actions_per_replan = validate_action_count(actions_per_replan)
    action_ids = list(action_ids)
    if not action_ids:
        return []

    if STOP_ACTION_ID in action_ids:
        action_ids = action_ids[:action_ids.index(STOP_ACTION_ID) + 1]

    return action_ids[:actions_per_replan]

def parse_action_sequence(output: str):
    if "</think>" in output.lower():
        output = re.split(r"</think>", output, flags=re.IGNORECASE)[-1]

    action_text = " ".join(output.split()).lower().strip(" \t\r\n`'\".,;:!?()[]{}")
    if not action_text:
        return []

    matches = []
    for action_id, pattern in ATOMIC_ACTION_PATTERNS:
        for match in pattern.finditer(action_text):
            matches.append((match.start(), match.end(), action_id))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    actions = []
    last_end = -1
    for start, end, action_id in matches:
        if start < last_end:
            continue
        actions.append(action_id)
        last_end = end
        if len(actions) >= VLN_ACTION_SEQUENCE_LENGTH:
            break

    return actions


def action_queue(text, *, actions_per_replan=4, stop_commit_max_actions=10,
                 uncertainty_actions=None, uncertainties=None,
                 uncertainty_budget=1.2, replan_action_range=(4, 8)):
    actions = parse_action_sequence(text)
    horizon = select_stop_commit_horizon(actions, stop_commit_max_actions)
    committed = horizon is not None
    if not committed:
        horizon = actions_per_replan
        if horizon == "uncertainty":
            minimum, maximum = validate_replan_action_range(replan_action_range)
            prefix = select_actions_for_replan(actions, maximum)
            if prefix != uncertainty_actions or uncertainties is None or len(uncertainties) != len(prefix):
                raise ValueError("Parsed actions and uncertainty records disagree")
            horizon = select_uncertainty_horizon(uncertainties, uncertainty_budget, (minimum, maximum))
    queue = select_actions_for_replan(actions, horizon) or [STOP_ACTION_ID]
    return actions, queue, {"horizon": horizon, "stop_committed": committed, "invalid_output": not actions}
