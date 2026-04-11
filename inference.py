from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.dirname(PACKAGE_DIR)
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from mobile_automation import MobileAutomationAction, MobileAutomationEnv

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
BENCHMARK = "quickcart"
TASKS = [("food_easy", 7, 12), ("food_medium", 7, 18), ("food_hard", 7, 25)]
SUCCESSFUL_LLM_CALLS = 0
ATTEMPTED_LLM_CALLS = 0
FIRST_PRE_LLM_ERROR: Optional[str] = None


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _format_number(value: float, decimals: int) -> str:
    if abs(value - 0.0) < 1e-9:
        return "0"
    if abs(value - 1.0) < 1e-9:
        return "1"
    return f"{value:.{decimals}f}"


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={_format_number(reward, 2)} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(_format_number(reward, 2) for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={_format_number(score, 3)} rewards={rewards_str}", flush=True)


def _format_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ").strip() or exc.__class__.__name__


def _record_pre_llm_error(task_id: str, stage: str, exc: Exception) -> None:
    global FIRST_PRE_LLM_ERROR
    if FIRST_PRE_LLM_ERROR is None:
        FIRST_PRE_LLM_ERROR = f"task={task_id} stage={stage} error={_format_error(exc)}"


async def build_env():
    if ENV_BASE_URL:
        return MobileAutomationEnv(base_url=ENV_BASE_URL)
    if IMAGE_NAME:
        return await MobileAutomationEnv.from_docker_image(IMAGE_NAME)
    raise RuntimeError("Missing IMAGE_NAME/LOCAL_IMAGE_NAME or ENV_BASE_URL.")


def fallback_action() -> MobileAutomationAction:
    return MobileAutomationAction(action_type="wait")


def _sanitize_payload_for_validation(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    action_type = sanitized.get("action_type")

    if action_type in {"back", "wait"}:
        sanitized["target_id"] = None
        sanitized["text"] = None
        sanitized["direction"] = None

    return sanitized


def _extract_action_payload(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Model returned empty content")
    try:
        return _sanitize_payload_for_validation(json.loads(raw_text))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise
        return _sanitize_payload_for_validation(json.loads(match.group(0)))


def _find_ui_value(observation, element_id: str) -> Optional[str]:
    for element in observation.ui_elements:
        if element.element_id == element_id:
            return element.value or element.text
    return None


def _cart_lines(observation) -> list[dict[str, Any]]:
    metadata_lines = observation.metadata.get("cart_snapshot") or []
    if metadata_lines:
        return metadata_lines

    lines: list[dict[str, Any]] = []
    for element in observation.ui_elements:
        if element.element_id.startswith("cart_line_"):
            lines.append(element.metadata)
    if lines:
        return lines

    review_summary = _find_ui_value(observation, "review_summary")
    if not review_summary:
        return lines

    for chunk in review_summary.split(","):
        part = chunk.strip()
        if " x" not in part:
            continue
        item_id, qty_text = part.rsplit(" x", 1)
        try:
            qty = int(qty_text.strip())
        except ValueError:
            continue
        lines.append({"item_id": item_id.strip(), "qty": qty, "customizations": {}})
    return lines


def _search_performed(observation) -> bool:
    if (value := _find_ui_value(observation, "search_bar")) and "spice" in value.lower():
        return True
    for entry in observation.recent_history:
        action = entry.action
        if action.get("action_type") == "type" and action.get("target_id") == "search_bar":
            if "spice" in (action.get("text") or "").lower():
                return True
    return False


def _observable_missing_requirements(observation) -> list[str]:
    lines = _cart_lines(observation)
    line_by_item = {line.get("item_id"): line for line in lines}
    metadata = observation.metadata or {}
    missing: list[str] = []

    if observation.task_id == "food_easy":
        paneer_line = line_by_item.get("paneer_wrap")
        if not paneer_line or paneer_line.get("qty") != 1 or paneer_line.get("customizations") != {}:
            missing.append("exactly one default Paneer Wrap")
        if observation.screen_id != "cart":
            missing.append("stop on cart screen")
        return missing

    if observation.task_id == "food_medium":
        if not _search_performed(observation):
            missing.append("search for Spice Route")
        paneer_line = line_by_item.get("paneer_wrap")
        biryani_line = line_by_item.get("veg_biryani")
        if not paneer_line or paneer_line.get("qty") != 1 or paneer_line.get("customizations") != {}:
            missing.append("Paneer Wrap x1")
        if not biryani_line or biryani_line.get("qty") != 1 or biryani_line.get("customizations") != {}:
            missing.append("Veg Biryani x1")
        if metadata.get("coupon_code") != "SAVE50" or metadata.get("discount_amount") != 50:
            missing.append("apply SAVE50")
        if observation.screen_id != "review_order":
            missing.append("stop on review order")
        return missing

    if observation.task_id == "food_hard":
        if not _search_performed(observation):
            missing.append("search for Spice Route")
        if (metadata.get("cart_total_qty") or 0) < 2:
            missing.append("quantity at least 2")
        if any(line.get("veg") is False for line in lines):
            missing.append("vegetarian cart only")
        if not any((line.get("customizations") or {}).get("no_onions") is True for line in lines):
            missing.append("at least one no_onions item")
        if (metadata.get("subtotal") or 0) > 400:
            missing.append("subtotal <= 400")
        if metadata.get("delivery_mode") != "no_contact":
            missing.append("no_contact delivery")
        if observation.screen_id != "review_order":
            missing.append("stop on review order")
        return missing

    return missing


def _observable_summary(observation) -> dict[str, Any]:
    return {
        "screen_id": observation.screen_id,
        "cart_lines": _cart_lines(observation),
        "coupon_input": _find_ui_value(observation, "coupon_input"),
        "delivery_mode": observation.metadata.get("delivery_mode"),
        "subtotal": observation.metadata.get("subtotal"),
        "cart_total_qty": observation.metadata.get("cart_total_qty"),
        "search_performed": _search_performed(observation),
        "missing_requirements": _observable_missing_requirements(observation),
    }


def _action_space_spec() -> dict[str, Any]:
    return {
        "allowed_action_types": ["tap", "type", "scroll", "back", "wait"],
        "rules": {
            "tap": "target_id is required; text must be null; direction must be null",
            "type": "target_id and text are required; direction must be null",
            "scroll": "direction is required and must be one of up/down/left/right; target_id must be null; text must be null",
            "back": "target_id must be null; text must be null; direction must be null",
            "wait": "target_id must be null; text must be null; direction must be null",
        },
        "forbidden_synonyms": ["click", "press", "input", "enter_text", "swipe"],
        "response_examples": [
            {"action_type": "tap", "target_id": "restaurant_card_spice_route", "text": None, "direction": None},
            {"action_type": "type", "target_id": "search_bar", "text": "spice", "direction": None},
            {"action_type": "scroll", "target_id": None, "text": None, "direction": "down"},
            {"action_type": "back", "target_id": None, "text": None, "direction": None},
            {"action_type": "wait", "target_id": None, "text": None, "direction": None},
        ],
        "response_contract": (
            "Return exactly one raw JSON object with keys action_type, target_id, text, and direction. "
            "Do not return markdown, code fences, comments, prose, or any extra keys."
        ),
    }


def model_action(client: OpenAI, observation) -> tuple[MobileAutomationAction, Optional[str]]:
    global ATTEMPTED_LLM_CALLS, SUCCESSFUL_LLM_CALLS
    prompt = {
        "goal": observation.goal,
        "current_state_summary": _observable_summary(observation),
        "screen_id": observation.screen_id,
        "last_action_status": observation.last_action_status,
        "recent_history": [entry.model_dump() for entry in observation.recent_history],
        "action_space": _action_space_spec(),
        "visible_target_ids": [element.element_id for element in observation.ui_elements if element.visible],
        "ui_elements": [element.model_dump() for element in observation.ui_elements],
        "xml_hierarchy": observation.xml_hierarchy,
        "screenshot_b64": observation.screenshot_b64,
        "instruction": (
            "Every clause in the goal is mandatory. "
            "Do not assume the task is complete just because you reached a plausible final screen. "
            "Use the current UI and recent history to verify search, cart contents, quantity, coupon, delivery mode, and budget before deciding to wait. "
            "Never place the order. "
            "Use only these exact action_type literals: tap, type, scroll, back, wait. "
            "Do not use synonyms such as click, press, input, enter_text, or swipe. "
            "Always return all four keys: action_type, target_id, text, direction. "
            "Use null for every unused field. "
            "Copy target_id values exactly from visible_target_ids."
        ),
    }
    try:
        ATTEMPTED_LLM_CALLS += 1
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are controlling QuickCart. "
                        "Choose exactly one safe next action based on the current observation and recent history. "
                        "Use the goal text, current UI, and recent history to decide whether the task is actually complete. "
                        "Return exactly one raw JSON object and no other text. "
                        "The JSON object must contain exactly these keys: action_type, target_id, text, direction."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0,
            stream=False,
        )
        SUCCESSFUL_LLM_CALLS += 1
        payload = _extract_action_payload(completion.choices[0].message.content or "")
        return MobileAutomationAction.model_validate(payload), None
    except Exception as exc:
        raise RuntimeError(f"LLM call failed on attempt {ATTEMPTED_LLM_CALLS}: {_format_error(exc)}") from exc


async def run_task(client: OpenAI, task_id: str, seed: int, max_steps: int) -> None:
    env = None
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    stage = "start"

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        stage = "build_env"
        env = await build_env()
        stage = "reset"
        result = await env.reset(task_id=task_id, seed=seed)
        stage = "post_reset"
        for step in range(1, max_steps + 1):
            if result.done:
                if ATTEMPTED_LLM_CALLS == 0:
                    _record_pre_llm_error(
                        task_id,
                        "reset_returned_done",
                        RuntimeError("Environment returned done=True before the first model action."),
                    )
                break
            stage = f"model_action_step_{step}"
            action, action_error = model_action(client, result.observation)
            stage = f"env_step_{step}"
            result = await env.step(action)
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            log_step(step, action.model_dump_json(), reward, result.done, action_error)
            if result.done:
                break
        score = float(result.observation.reward_breakdown.final_score)
        success = score >= 0.99 - 1e-9
    except Exception as exc:
        if ATTEMPTED_LLM_CALLS == 0:
            _record_pre_llm_error(task_id, stage, exc)
        step_no = steps_taken + 1 if steps_taken > 0 else 0
        log_step(step_no, "null", 0.0, True, _format_error(exc))
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    if not API_KEY or not API_BASE_URL:
        missing = []
        if not API_KEY:
            missing.append("API_KEY")
        if not API_BASE_URL:
            missing.append("API_BASE_URL")
        error_message = f"Missing required environment variable(s): {', '.join(missing)}."
        for task_id, _seed, _max_steps in TASKS:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_step(0, "null", 0.0, True, error_message)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id, seed, max_steps in TASKS:
        try:
            await run_task(client, task_id, seed, max_steps)
        except Exception as exc:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_step(0, "null", 0.0, True, _format_error(exc))
            log_end(success=False, steps=0, score=0.0, rewards=[])

    if ATTEMPTED_LLM_CALLS == 0:
        if FIRST_PRE_LLM_ERROR is not None:
            raise RuntimeError(
                "No LLM calls were attempted. "
                f"First pre-LLM failure: {FIRST_PRE_LLM_ERROR}"
            )
        raise RuntimeError("No LLM calls were attempted")
    if SUCCESSFUL_LLM_CALLS == 0:
        raise RuntimeError("LLM calls were attempted but none succeeded.")


if __name__ == "__main__":
    asyncio.run(main())
