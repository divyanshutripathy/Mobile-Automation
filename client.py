# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mobile Automation Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MobileAutomationAction, MobileAutomationObservation


class MobileAutomationEnv(
    EnvClient[MobileAutomationAction, MobileAutomationObservation, State]
):
    """
    Client for the Mobile Automation Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MobileAutomationEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.screen_id)
        ...
        ...     result = client.step(MobileAutomationAction(action_type="wait"))
        ...     print(result.observation.last_action_status)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MobileAutomationEnv.from_docker_image("mobile_automation:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MobileAutomationAction(action_type="wait"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MobileAutomationAction) -> Dict:
        """
        Convert MobileAutomationAction to JSON payload for step message.

        Args:
            action: MobileAutomationAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {"action_type": action.action_type}
        if action.target_id is not None:
            payload["target_id"] = action.target_id
        if action.text is not None:
            payload["text"] = action.text
        if action.direction is not None:
            payload["direction"] = action.direction
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[MobileAutomationObservation]:
        """
        Parse server response into StepResult[MobileAutomationObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MobileAutomationObservation
        """
        obs_data = payload.get("observation", {})
        observation = MobileAutomationObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", obs_data.get("done", False)),
                "reward": payload.get("reward", obs_data.get("reward", 0.0)),
            }
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
