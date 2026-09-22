from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from puripuly_heart.core.clock import Clock


@dataclass(frozen=True, slots=True)
class HealthChallenge:
    challenge_id: int
    payload: dict[str, Any]


class AuthenticatedSessionHealth:
    def __init__(
        self,
        *,
        overlay_instance_id: str | None,
        runtime_generation: int,
        clock: Clock,
    ) -> None:
        self.overlay_instance_id = overlay_instance_id
        self.runtime_generation = runtime_generation
        self._clock = clock
        self.challenges: OrderedDict[int, float] = OrderedDict()
        self.next_challenge_id = 1
        self.next_challenge_at = 0.0
        self.owner_health_deadline: float | None = None
        self.native_acceptance_revision: int | None = None
        self.native_acceptance_deadline: float | None = None
        self.failure_reported = False
        self.token_consumed = False

    def begin(self) -> None:
        now = self._clock.now()
        if self.overlay_instance_id is not None:
            self.next_challenge_at = now
            self.owner_health_deadline = now + 3.0
        self.failure_reported = False

    def validate_auth(
        self,
        payload: Mapping[str, Any],
        *,
        session_token: str,
        token_consumed: bool,
        stopping: bool,
        desktop_runtime: bool,
        contract_version: int,
        execution_contract: Mapping[str, Any],
        native_retry_contract: Mapping[str, Any],
        speaker_transition_contract: Mapping[str, Any],
    ) -> bool:
        if (
            payload.get("type") != "auth"
            or payload.get("session_token") != session_token
            or token_consumed
            or stopping
        ):
            return False
        if self.overlay_instance_id is None:
            return True
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, dict):
            return False
        if capabilities.get("execution_contract") != execution_contract:
            return False
        if capabilities.get("speaker_transition_presentation") != speaker_transition_contract:
            return False
        if (
            not desktop_runtime
            and capabilities.get("native_presentation_retry") != native_retry_contract
        ):
            return False
        return (
            payload.get("contract_version") == contract_version
            and payload.get("overlay_instance_id") == self.overlay_instance_id
            and payload.get("runtime_generation") == self.runtime_generation
        )

    def issue_challenge_if_due(self) -> HealthChallenge | None:
        if self.overlay_instance_id is None:
            return None
        now = self._clock.now()
        if now < self.next_challenge_at:
            return None
        challenge_id = self.next_challenge_id
        self.next_challenge_id += 1
        self.record_challenge(challenge_id, now)
        self.next_challenge_at = now + 1.0
        return HealthChallenge(
            challenge_id=challenge_id,
            payload={
                "type": "health_challenge",
                "challenge_id": challenge_id,
                "overlay_instance_id": self.overlay_instance_id,
                "runtime_generation": self.runtime_generation,
            },
        )

    def record_challenge(self, challenge_id: int, issued_at: float) -> None:
        self.challenges[challenge_id] = issued_at
        while len(self.challenges) > 4:
            self.challenges.popitem(last=False)

    def record_scene_written(self, scene_revision: int) -> None:
        self.native_acceptance_revision = max(
            scene_revision,
            self.native_acceptance_revision or scene_revision,
        )
        if self.native_acceptance_deadline is None:
            self.native_acceptance_deadline = self._clock.now() + 2.0

    def failure_due(self) -> str | None:
        now = self._clock.now()
        cause = None
        if self.native_acceptance_deadline is not None and now >= self.native_acceptance_deadline:
            cause = "native_acceptance_timeout"
        elif self.owner_health_deadline is not None and now >= self.owner_health_deadline:
            cause = "native_owner_unresponsive"
        if cause is None or self.failure_reported:
            return None
        self.failure_reported = True
        return cause

    def handle_owner_status(self, message: Mapping[str, Any]) -> dict[str, Any]:
        if (
            message.get("overlay_instance_id") != self.overlay_instance_id
            or message.get("runtime_generation") != self.runtime_generation
        ):
            raise ValueError("invalid owner status identity")
        challenge_id = message.get("health_challenge_id")
        now = self._clock.now()
        valid_response = False
        issued_at: float | None = None
        if isinstance(challenge_id, int) and not isinstance(challenge_id, bool):
            issued_at = self.challenges.pop(challenge_id, None)
            if issued_at is not None and now <= issued_at + 3.0:
                valid_response = True
                self.owner_health_deadline = issued_at + 3.0
                for prior in tuple(self.challenges):
                    if prior <= challenge_id:
                        self.challenges.pop(prior, None)
        applied_revision = message.get("latest_applied_revision")
        if (
            valid_response
            and isinstance(applied_revision, int)
            and not isinstance(applied_revision, bool)
            and self.native_acceptance_revision is not None
            and applied_revision >= self.native_acceptance_revision
        ):
            self.native_acceptance_revision = None
            self.native_acceptance_deadline = None
        forwarded = dict(message)
        forwarded["health_challenge_validated"] = valid_response
        forwarded["health_challenge_issued_at"] = issued_at if valid_response else None
        return forwarded
