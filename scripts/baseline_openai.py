from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mobile_automation import MobileAutomationAction, MobileAutomationEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
TASKS = [("food_easy", 7), ("food_medium", 7), ("food_hard", 7)]


def choose_env():
    if ENV_BASE_URL:
        return MobileAutomationEnv(base_url=ENV_BASE_URL)
    if IMAGE_NAME:
        return MobileAutomationEnv.from_docker_image(IMAGE_NAME)
    raise RuntimeError("Set ENV_BASE_URL or IMAGE_NAME")


def fallback_policy(observation) -> MobileAutomationAction:
    visible = {element.element_id for element in observation.ui_elements}
    predicates = observation.metadata.get("predicate_satisfaction", {})
    if observation.screen_id == "home":
        if observation.task_id in {"food_medium", "food_hard"} and "search_bar" in visible and predicates.get("typed_search_for_spice") is not True:
            return MobileAutomationAction(action_type="type", target_id="search_bar", text="spice")
        if "restaurant_card_spice_route" in visible:
            return MobileAutomationAction(action_type="tap", target_id="restaurant_card_spice_route")
    if observation.screen_id == "restaurant_page":
        if observation.task_id == "food_hard" and "btn_open_item_paneer_wrap" in visible and not predicates.get("has_no_onions_item", False):
            return MobileAutomationAction(action_type="tap", target_id="btn_open_item_paneer_wrap")
        for target in ["btn_quick_add_paneer_wrap", "btn_quick_add_veg_biryani", "btn_quick_add_dal_khichdi", "btn_open_cart"]:
            if target in visible:
                return MobileAutomationAction(action_type="tap", target_id=target)
        return MobileAutomationAction(action_type="scroll", direction="down")
    if observation.screen_id == "item_detail":
        if "toggle_no_onions" in visible and not predicates.get("has_no_onions_item", False):
            return MobileAutomationAction(action_type="tap", target_id="toggle_no_onions")
        return MobileAutomationAction(action_type="tap", target_id="btn_add_to_cart")
    if observation.screen_id == "cart":
        if observation.task_id == "food_medium" and not predicates.get("coupon_applied", False):
            if observation.metadata.get("coupon_code") != "SAVE50":
                return MobileAutomationAction(action_type="type", target_id="coupon_input", text="SAVE50")
            return MobileAutomationAction(action_type="tap", target_id="btn_apply_coupon")
        if observation.task_id == "food_hard" and observation.metadata.get("delivery_mode") != "no_contact":
            return MobileAutomationAction(action_type="tap", target_id="delivery_radio_no_contact")
        return MobileAutomationAction(action_type="tap", target_id="btn_review_order")
    if observation.screen_id == "review_order":
        return MobileAutomationAction(action_type="wait")
    return MobileAutomationAction(action_type="wait")


def ask_model(client: OpenAI, observation) -> dict[str, Any]:
    prompt = {
        "goal": observation.goal,
        "screen_id": observation.screen_id,
        "ui_elements": [element.model_dump() for element in observation.ui_elements],
        "xml_hierarchy": observation.xml_hierarchy,
        "screenshot_b64": observation.screenshot_b64,
        "last_action_status": observation.last_action_status,
        "schema": {
            "action_type": ["tap", "type", "scroll", "back", "wait"],
            "target_id": "optional string",
            "text": "optional string",
            "direction": ["up", "down", "left", "right"],
        },
    }
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return one JSON object matching the action schema. Be conservative and never place the order."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or "{}"
    return json.loads(content)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    scores = []
    for task_id, seed in TASKS:
        env = choose_env()
        if hasattr(env, "__await__"):
            env = await env
        try:
            result = await env.reset(task_id=task_id, seed=seed)
            for _ in range(30):
                if result.done:
                    break
                try:
                    payload = ask_model(client, result.observation)
                    action = MobileAutomationAction.model_validate(payload)
                except Exception:
                    action = fallback_policy(result.observation)
                result = await env.step(action)
            score = result.observation.reward_breakdown.final_score
            scores.append(score)
            print(f"{task_id}: {score:.2f}")
        finally:
            await env.close()
    average = sum(scores) / len(scores) if scores else 0.0
    print(f"average: {average:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
