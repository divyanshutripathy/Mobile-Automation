"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    OPENAI_API_KEY The OpenAI API key to use for inference.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-mini")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.dirname(PACKAGE_DIR)
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from mobile_automation import MobileAutomationAction, MobileAutomationEnv

LOCAL_IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-5.4-mini"
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
BENCHMARK = "quickcart"
TASKS = [("food_easy", 7, 12), ("food_medium", 7, 18), ("food_hard", 7, 25)]

ACTION_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit_action",
            "description": "Choose the next QuickCart action. Never place the order unless explicitly instructed.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["tap", "type", "scroll", "back", "wait"],
                    },
                    "target_id": {"type": ["string", "null"]},
                    "text": {"type": ["string", "null"]},
                    "direction": {
                        "type": ["string", "null"],
                        "enum": ["up", "down", "left", "right", None],
                    },
                },
                "required": ["action_type", "target_id", "text", "direction"],
            },
        },
    }
]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


async def build_env():
    if ENV_BASE_URL:
        return MobileAutomationEnv(base_url=ENV_BASE_URL)
    return await MobileAutomationEnv.from_docker_image(LOCAL_IMAGE_NAME)


def fallback_action() -> MobileAutomationAction:
    return MobileAutomationAction(action_type="wait")


def _extract_tool_args(message: Any) -> dict[str, Any]:
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        raise ValueError(f"No tool call returned. Raw content: {getattr(message, 'content', None)!r}")
    arguments = tool_calls[0].function.arguments
    if isinstance(arguments, dict):
        return arguments
    return json.loads(arguments)


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
    summary = {
        "screen_id": observation.screen_id,
        "cart_lines": _cart_lines(observation),
        "coupon_input": _find_ui_value(observation, "coupon_input"),
        "delivery_mode": observation.metadata.get("delivery_mode"),
        "subtotal": observation.metadata.get("subtotal"),
        "cart_total_qty": observation.metadata.get("cart_total_qty"),
        "search_performed": _search_performed(observation),
        "missing_requirements": _observable_missing_requirements(observation),
    }
    return summary


def model_action(client: OpenAI, observation) -> tuple[MobileAutomationAction, Optional[str]]:
    prompt = {
        "goal": observation.goal,
        "current_state_summary": _observable_summary(observation),
        "screen_id": observation.screen_id,
        "last_action_status": observation.last_action_status,
        "recent_history": [entry.model_dump() for entry in observation.recent_history],
        "ui_elements": [element.model_dump() for element in observation.ui_elements],
        "xml_hierarchy": observation.xml_hierarchy,
        "screenshot_b64": observation.screenshot_b64,
        "instruction": (
            "Every clause in the goal is mandatory. "
            "Do not assume the task is complete just because you reached a plausible final screen. "
            "Use the current UI and recent history to verify search, cart contents, quantity, coupon, delivery mode, and budget before deciding to wait. "
            "Never place the order."
        ),
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are controlling QuickCart. "
                        "Choose exactly one safe next action based on the current observation and recent history. "
                        "Use the goal text, current UI, and recent history to decide whether the task is actually complete. "
                        "Use the submit_action function for every reply."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0,
            max_completion_tokens=200,
            tools=ACTION_TOOL,
            tool_choice={"type": "function", "function": {"name": "submit_action"}},
            stream=False,
        )
        payload = _extract_tool_args(completion.choices[0].message)
        return MobileAutomationAction.model_validate(payload), None
    except Exception as exc:
        return fallback_action(), str(exc).replace("\n", " ")


async def run_task(client: OpenAI, task_id: str, seed: int, max_steps: int) -> None:
    env = await build_env()
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        result = await env.reset(task_id=task_id, seed=seed)
        for step in range(1, max_steps + 1):
            if result.done:
                break
            action, action_error = model_action(client, result.observation)
            result = await env.step(action)
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            log_step(step, action.model_dump_json(), reward, result.done, action_error)
            if result.done:
                break
        score = float(result.observation.reward_breakdown.final_score)
        success = score >= 1.0 - 1e-9
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    if not API_KEY:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or API_KEY in your .env file.")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id, seed, max_steps in TASKS:
        await run_task(client, task_id, seed, max_steps)


if __name__ == "__main__":
    asyncio.run(main())
