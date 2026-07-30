from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from puripuly_heart.core.runtime.github_star_prompt import GithubStarPromptRuntime

GITHUB_STAR_PROMPT_ELIGIBLE_LAUNCH_THRESHOLD = 3
GITHUB_STAR_PROMPT_RECENCY_WINDOW = timedelta(days=14)


class GithubStarPromptUiState(Protocol):
    github_star_prompt_clicked: bool
    github_star_prompt_last_shown_at: str | None
    github_star_prompt_show_count: int
    github_star_prompt_translation_success_observed: bool
    github_star_prompt_eligible_launch_count: int


class GithubStarPromptSettings(Protocol):
    ui: GithubStarPromptUiState


def github_star_prompt_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def github_star_prompt_utc_timestamp(value: datetime | None = None) -> str:
    resolved = value or github_star_prompt_utc_now()
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    return (
        resolved.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def parse_github_star_prompt_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    parse_value = f"{normalized[:-1]}+00:00" if normalized.endswith("Z") else normalized
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed.astimezone(timezone.utc)


def github_star_prompt_non_negative_count(value: object) -> int:
    if type(value) is int and value >= 0:
        return value
    return 0


def github_star_prompt_latest_timestamp(*values: str | None) -> str | None:
    latest: tuple[datetime, str] | None = None
    for value in values:
        parsed = parse_github_star_prompt_timestamp(value)
        if parsed is None:
            continue
        normalized_value = github_star_prompt_utc_timestamp(parsed)
        if latest is None or parsed > latest[0]:
            latest = (parsed, normalized_value)
    return latest[1] if latest is not None else None


@dataclass(slots=True)
class GithubStarPromptOwner:
    settings_provider: Callable[[], GithubStarPromptSettings | None]
    persist_settings_state: Callable[
        [GithubStarPromptSettings, GithubStarPromptSettings],
        Awaitable[bool],
    ]
    is_eligible: Callable[[], bool]
    has_user_owned_cloud_connection: Callable[[GithubStarPromptSettings | None], bool]
    log_save_failure: Callable[[str, Exception], None]
    runtime_diagnostics_sink: Callable[[str, Mapping[str, object]], None]
    translation_success_observation: Callable[[], Coroutine[Any, Any, bool]] | None = None
    _runtime: GithubStarPromptRuntime | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _persistence_lock: asyncio.Lock | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def runtime(self) -> GithubStarPromptRuntime | None:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: GithubStarPromptRuntime | None) -> None:
        self._runtime = runtime

    @property
    def persistence_lock(self) -> asyncio.Lock:
        if self._persistence_lock is None:
            self._persistence_lock = asyncio.Lock()
        return self._persistence_lock

    def get_runtime(self) -> GithubStarPromptRuntime:
        if self._runtime is None:
            self._runtime = GithubStarPromptRuntime(
                diagnostics_sink=self.runtime_diagnostics_sink,
            )
        return self._runtime

    def initial_launch_gate_satisfied(self, settings: GithubStarPromptSettings) -> bool:
        ui = settings.ui
        if github_star_prompt_non_negative_count(ui.github_star_prompt_show_count) > 0:
            return True
        return (
            github_star_prompt_non_negative_count(ui.github_star_prompt_eligible_launch_count)
            >= GITHUB_STAR_PROMPT_ELIGIBLE_LAUNCH_THRESHOLD
        )

    def should_show(self, *, now: datetime | None = None) -> bool:
        settings = self.settings_provider()
        if settings is None:
            return False
        ui = settings.ui
        if ui.github_star_prompt_clicked:
            return False
        if not self.is_eligible():
            return False
        if not self.initial_launch_gate_satisfied(settings):
            return False
        last_shown_at = parse_github_star_prompt_timestamp(ui.github_star_prompt_last_shown_at)
        if last_shown_at is None:
            return True
        resolved_now = now or github_star_prompt_utc_now()
        if resolved_now.tzinfo is None:
            resolved_now = resolved_now.replace(tzinfo=timezone.utc)
        elapsed = resolved_now.astimezone(timezone.utc) - last_shown_at
        return elapsed >= GITHUB_STAR_PROMPT_RECENCY_WINDOW

    async def persist_mutation(
        self,
        *,
        failure_context: str,
        mutate: Callable[[GithubStarPromptSettings], bool],
    ) -> bool:
        attempted_mutation = False
        while True:
            async with self.persistence_lock:
                settings = self.settings_provider()
                if settings is None:
                    return False
                snapshot = self.state_snapshot(settings)
                base_settings = copy.deepcopy(settings)
                if not mutate(settings):
                    return attempted_mutation
                attempted_mutation = True
                try:
                    persisted = await self.persist_settings_state(
                        base_settings,
                        settings,
                    )
                except asyncio.CancelledError:
                    if self.settings_provider() is settings:
                        self.restore_state_snapshot(settings, snapshot)
                    raise
                except Exception as exc:
                    if self.settings_provider() is settings:
                        self.restore_state_snapshot(settings, snapshot)
                    self.log_save_failure(failure_context, exc)
                    return False
                if not persisted:
                    if self.settings_provider() is settings:
                        self.restore_state_snapshot(settings, snapshot)
                    return False
                if self.settings_provider() is settings:
                    return True
            await asyncio.sleep(0)

    async def persist_opened(
        self,
        *,
        opened_at: datetime | None = None,
        should_open: Callable[[], bool] | None = None,
    ) -> bool:
        opened_timestamp = github_star_prompt_utc_timestamp(opened_at)
        while True:
            async with self.persistence_lock:
                settings = self.settings_provider()
                if settings is None:
                    return False
                if should_open is not None and not should_open():
                    return False
                snapshot = self.state_snapshot(settings)
                base_settings = copy.deepcopy(settings)
                ui = settings.ui
                ui.github_star_prompt_last_shown_at = opened_timestamp
                ui.github_star_prompt_show_count = (
                    github_star_prompt_non_negative_count(ui.github_star_prompt_show_count) + 1
                )
                try:
                    persisted = await self.persist_settings_state(
                        base_settings,
                        settings,
                    )
                except asyncio.CancelledError:
                    if self.settings_provider() is settings:
                        self.restore_state_snapshot(settings, snapshot)
                    raise
                except Exception as exc:
                    if self.settings_provider() is settings:
                        self.restore_state_snapshot(settings, snapshot)
                    self.log_save_failure("open state", exc)
                    return False
                if not persisted:
                    if self.settings_provider() is settings:
                        self.restore_state_snapshot(settings, snapshot)
                    return False
                if self.settings_provider() is settings:
                    if should_open is not None and not should_open():
                        rollback_base_settings = copy.deepcopy(settings)
                        self.restore_state_snapshot(settings, snapshot)
                        try:
                            await self.persist_settings_state(
                                rollback_base_settings,
                                settings,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            self.log_save_failure("open state rollback", exc)
                        return False
                    return True
            await asyncio.sleep(0)

    async def persist_eligible_launch(self) -> bool:
        settings = self.settings_provider()
        if settings is None:
            return False
        ui = settings.ui
        if ui.github_star_prompt_clicked:
            return False
        if not self.is_eligible():
            return False
        if self.initial_launch_gate_satisfied(settings):
            return True

        def mutate(current: GithubStarPromptSettings) -> bool:
            current_ui = current.ui
            if current_ui.github_star_prompt_clicked:
                return False
            if not self.is_eligible():
                return False
            if self.initial_launch_gate_satisfied(current):
                return False
            current_count = github_star_prompt_non_negative_count(
                current_ui.github_star_prompt_eligible_launch_count
            )
            current_ui.github_star_prompt_eligible_launch_count = min(
                current_count + 1,
                GITHUB_STAR_PROMPT_ELIGIBLE_LAUNCH_THRESHOLD,
            )
            return True

        await self.persist_mutation(
            failure_context="eligible launch state",
            mutate=mutate,
        )
        settings = self.settings_provider()
        return bool(
            settings is not None
            and not settings.ui.github_star_prompt_clicked
            and self.is_eligible()
            and self.initial_launch_gate_satisfied(settings)
        )

    async def persist_clicked(self) -> bool:
        def mutate(settings: GithubStarPromptSettings) -> bool:
            settings.ui.github_star_prompt_clicked = True
            return True

        return await self.persist_mutation(
            failure_context="click state",
            mutate=mutate,
        )

    async def persist_translation_success_observed(self) -> bool:
        def mutate(settings: GithubStarPromptSettings) -> bool:
            if not self.has_user_owned_cloud_connection(settings):
                return False
            ui = settings.ui
            if ui.github_star_prompt_translation_success_observed:
                return False
            ui.github_star_prompt_translation_success_observed = True
            return True

        return await self.persist_mutation(
            failure_context="translation success observation",
            mutate=mutate,
        )

    def record_opened(self, *, opened_at: datetime | None = None) -> bool:
        return self.run_sync(self.persist_opened(opened_at=opened_at))

    def record_clicked(self) -> bool:
        return self.run_sync(self.persist_clicked())

    def record_translation_success_observed(self) -> bool:
        return self.run_sync(self.persist_translation_success_observed())

    def schedule_translation_success_observed(self) -> bool:
        settings = self.settings_provider()
        if settings is None:
            return False
        if not self.has_user_owned_cloud_connection(settings):
            return False
        if settings.ui.github_star_prompt_translation_success_observed:
            return False
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self.record_translation_success_observed()
        runtime = self.get_runtime()
        existing_task = runtime.translation_success_task
        if existing_task is not None and not existing_task.done():
            return False
        try:
            observation = (
                self.translation_success_observation()
                if self.translation_success_observation is not None
                else self.persist_translation_success_observed()
            )
            runtime.start_translation_success_observation(observation)
        except RuntimeError:
            return False
        return True

    async def drain_translation_success_observation(self) -> None:
        if self._runtime is not None:
            await self._runtime.drain_translation_success_task()

    async def preserve_before_settings_replace(
        self,
        replacement_settings: GithubStarPromptSettings,
    ) -> None:
        await self.drain_translation_success_observation()
        async with self.persistence_lock:
            current_settings = self.settings_provider()
            if current_settings is None:
                return
            current_ui = current_settings.ui
            replacement_ui = replacement_settings.ui
            replacement_ui.github_star_prompt_clicked = bool(
                replacement_ui.github_star_prompt_clicked or current_ui.github_star_prompt_clicked
            )
            replacement_ui.github_star_prompt_translation_success_observed = bool(
                replacement_ui.github_star_prompt_translation_success_observed
                or current_ui.github_star_prompt_translation_success_observed
            )
            replacement_ui.github_star_prompt_eligible_launch_count = max(
                github_star_prompt_non_negative_count(
                    replacement_ui.github_star_prompt_eligible_launch_count
                ),
                github_star_prompt_non_negative_count(
                    current_ui.github_star_prompt_eligible_launch_count
                ),
            )
            replacement_ui.github_star_prompt_show_count = max(
                github_star_prompt_non_negative_count(replacement_ui.github_star_prompt_show_count),
                github_star_prompt_non_negative_count(current_ui.github_star_prompt_show_count),
            )
            replacement_ui.github_star_prompt_last_shown_at = github_star_prompt_latest_timestamp(
                replacement_ui.github_star_prompt_last_shown_at,
                current_ui.github_star_prompt_last_shown_at,
            )

    def stop_ingress(self) -> None:
        if self._runtime is not None:
            self._runtime.stop_ingress()

    async def close(self) -> None:
        if self._runtime is not None:
            await self._runtime.close()

    @staticmethod
    def run_sync(coroutine: Any) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return bool(asyncio.run(coroutine))
        close = getattr(coroutine, "close", None)
        if callable(close):
            close()
        return False

    @staticmethod
    def state_snapshot(settings: GithubStarPromptSettings) -> tuple[object, ...]:
        ui = settings.ui
        return (
            ui.github_star_prompt_clicked,
            ui.github_star_prompt_last_shown_at,
            ui.github_star_prompt_show_count,
            ui.github_star_prompt_translation_success_observed,
            ui.github_star_prompt_eligible_launch_count,
        )

    @staticmethod
    def restore_state_snapshot(
        settings: GithubStarPromptSettings,
        snapshot: tuple[object, ...],
    ) -> None:
        (
            clicked,
            last_shown_at,
            show_count,
            translation_success_observed,
            eligible_launch_count,
        ) = snapshot
        ui = settings.ui
        ui.github_star_prompt_clicked = bool(clicked)
        ui.github_star_prompt_last_shown_at = (
            last_shown_at if isinstance(last_shown_at, str) else None
        )
        ui.github_star_prompt_show_count = github_star_prompt_non_negative_count(show_count)
        ui.github_star_prompt_translation_success_observed = bool(translation_success_observed)
        ui.github_star_prompt_eligible_launch_count = github_star_prompt_non_negative_count(
            eligible_launch_count
        )


__all__ = [
    "GITHUB_STAR_PROMPT_ELIGIBLE_LAUNCH_THRESHOLD",
    "GITHUB_STAR_PROMPT_RECENCY_WINDOW",
    "GithubStarPromptOwner",
    "GithubStarPromptSettings",
    "GithubStarPromptUiState",
    "github_star_prompt_latest_timestamp",
    "github_star_prompt_non_negative_count",
    "github_star_prompt_utc_now",
    "github_star_prompt_utc_timestamp",
    "parse_github_star_prompt_timestamp",
]
