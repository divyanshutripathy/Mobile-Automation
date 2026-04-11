from __future__ import annotations

try:
    from ..models import RewardBreakdown
    from .sim_state import SimState, state_hash, subtotal, total_after_discount, total_qty
    from .tasks import TaskSpec
except ImportError:
    from models import RewardBreakdown
    from server.sim_state import SimState, state_hash, subtotal, total_after_discount, total_qty
    from server.tasks import TaskSpec

def clamp_public_score(value: float, *, allow_zero: bool = False) -> float:
    if allow_zero and value <= 0.0:
        return 0.0
    return max(0.01, min(0.99, value))

def predicate_satisfaction(state: SimState, task_spec: TaskSpec, terminal: bool = False) -> dict[str, bool]:
    predicates = task_spec.terminal_predicates if terminal else task_spec.progress_predicates
    return {predicate.name: predicate.predicate_fn(state) for predicate in predicates}


def compute_progress_score(state: SimState, task_spec: TaskSpec) -> tuple[float, dict[str, bool]]:
    satisfied = predicate_satisfaction(state, task_spec, terminal=False)
    total_weight = sum(predicate.weight for predicate in task_spec.progress_predicates)
    score = 0.0
    for predicate in task_spec.progress_predicates:
        if satisfied[predicate.name]:
            score += predicate.weight
    normalized = max(0.0, min(1.0, score / total_weight)) if total_weight else 0.0
    if abs(normalized - 1.0) < 1e-9:
        normalized = 1.0
    return normalized, satisfied


def is_success(state: SimState, task_spec: TaskSpec) -> tuple[bool, dict[str, bool]]:
    satisfied = predicate_satisfaction(state, task_spec, terminal=True)
    return all(satisfied.values()), satisfied


def build_metadata(state: SimState, task_spec: TaskSpec, progress_satisfied: dict[str, bool], terminal_satisfied: dict[str, bool]) -> dict:
    for name, hit in progress_satisfied.items():
        if hit:
            state.checkpoints_hit.add(name)
    return {
        "task_difficulty": task_spec.difficulty,
        "predicate_satisfaction": progress_satisfied,
        "terminal_predicates": terminal_satisfied,
        "checkpoints_hit": sorted(state.checkpoints_hit),
        "cart_snapshot": [dict(line) for line in state.cart],
        "subtotal": subtotal(state),
        "discount_amount": state.discount_amount,
        "total": total_after_discount(state),
        "delivery_mode": state.delivery_mode,
        "coupon_code": state.coupon_code,
        "cart_total_qty": total_qty(state),
        "selected_restaurant_id": state.selected_restaurant_id,
        "state_hash": state_hash(state),
    }


def compute_reward(prev_progress: float, new_progress: float, action_status: str, state: SimState, current_action_key: str | None = None) -> RewardBreakdown:
    delta_progress = new_progress - prev_progress
    step_penalty = 0.01
    invalid_penalty = 0.03 if action_status == "invalid" else 0.0
    loop_penalty = 0.02 if len(state.recent_state_hashes) >= 2 and state.recent_state_hashes[-1] in state.recent_state_hashes[-4:-1] else 0.0
    repetition_penalty = 0.0
    if current_action_key is not None and state.recent_action_keys and delta_progress <= 0:
        current_action = current_action_key
        consecutive_repeats = 0
        for action_key in reversed(state.recent_action_keys):
            if action_key != current_action:
                break
            consecutive_repeats += 1
        if consecutive_repeats >= 2:
            repetition_penalty = 0.05
        elif consecutive_repeats >= 1:
            repetition_penalty = 0.03
    forbidden_penalty = 1.0 if action_status == "forbidden" else 0.0
    final_score = 0.0 if state.forbidden_triggered else new_progress
    return RewardBreakdown(
        progress_score=0.0 if state.forbidden_triggered else clamp_public_score(new_progress),
        delta_progress=0.0 if action_status == "forbidden" else delta_progress,
        step_penalty=step_penalty,
        invalid_penalty=invalid_penalty,
        loop_penalty=loop_penalty,
        repetition_penalty=repetition_penalty,
        forbidden_penalty=forbidden_penalty,
        final_score=clamp_public_score(final_score, allow_zero=state.forbidden_triggered),
    )
