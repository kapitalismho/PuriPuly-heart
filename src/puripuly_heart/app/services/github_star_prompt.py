from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, GithubStarPromptState
from puripuly_heart.core.runtime.github_star_prompt import GithubStarPromptRuntime

GITHUB_STAR_PROMPT_ELIGIBLE_LAUNCH_THRESHOLD = 3
GITHUB_STAR_PROMPT_RECENCY_WINDOW = timedelta(days=14)


class GithubStarPromptUiState(Protocol):
    clicked: bool
    last_shown_at: str | None
    show_count: int
    translation_success_observed: bool
    eligible_launch_count: int


class GithubStarPromptSettings(Protocol):
    state: object


def _star_state(settings: object) -> GithubStarPromptState:
    return settings.state.github_star_prompt


def _with_star_state(settings: AppSettingsVNext, **changes: object) -> AppSettingsVNext:
    return replace(
        settings,
        state=replace(
            settings.state,
            github_star_prompt=replace(settings.state.github_star_prompt, **changes),
        ),
    )


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
        prompt = _star_state(settings)
        if github_star_prompt_non_negative_count(prompt.show_count) > 0:
            return True
        return (
            github_star_prompt_non_negative_count(prompt.eligible_launch_count)
            >= GITHUB_STAR_PROMPT_ELIGIBLE_LAUNCH_THRESHOLD
        )

    def should_show(self, *, now: datetime | None = None) -> bool:
        settings = self.settings_provider()
        if settings is None:
            return False
        prompt = _star_state(settings)
        if prompt.clicked:
            return False
        if not self.is_eligible():
            return False
        if not self.initial_launch_gate_satisfied(settings):
            return False
        last_shown_at = parse_github_star_prompt_timestamp(prompt.last_shown_at)
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
        mutate: Callable[[AppSettingsVNext], AppSettingsVNext | None],
    ) -> bool:
        async with self.persistence_lock:
            settings = self.settings_provider()
            if settings is None or not isinstance(settings, AppSettingsVNext):
                return False
            base_settings = copy.deepcopy(settings)
            updated = mutate(copy.deepcopy(settings))
            if updated is None:
                return False
            try:
                return await self.persist_settings_state(base_settings, updated)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log_save_failure(failure_context, exc)
                return False

    async def persist_opened(
        self,
        *,
        opened_at: datetime | None = None,
        should_open: Callable[[], bool] | None = None,
    ) -> bool:
        opened_timestamp = github_star_prompt_utc_timestamp(opened_at)
        async with self.persistence_lock:
            settings = self.settings_provider()
            if settings is None or not isinstance(settings, AppSettingsVNext):
                return False
            if should_open is not None and not should_open():
                return False
            snapshot = self.state_snapshot(settings)
            base_settings = copy.deepcopy(settings)
            prompt = _star_state(settings)
            updated = _with_star_state(
                copy.deepcopy(settings),
                last_shown_at=opened_timestamp,
                show_count=github_star_prompt_non_negative_count(prompt.show_count) + 1,
            )
            try:
                persisted = await self.persist_settings_state(base_settings, updated)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log_save_failure("open state", exc)
                return False
            if not persisted:
                return False
            live = self.settings_provider()
            if should_open is not None and not should_open():
                if live is None or not isinstance(live, AppSettingsVNext):
                    return False
                restored = self.restore_state_snapshot(live, snapshot)
                try:
                    await self.persist_settings_state(live, restored)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.log_save_failure("open state rollback", exc)
                return False
            return True

    async def persist_eligible_launch(self) -> bool:
        settings = self.settings_provider()
        if settings is None:
            return False
        prompt = _star_state(settings)
        if prompt.clicked:
            return False
        if not self.is_eligible():
            return False
        if self.initial_launch_gate_satisfied(settings):
            return True

        def mutate(current: AppSettingsVNext) -> AppSettingsVNext | None:
            current_prompt = _star_state(current)
            if current_prompt.clicked:
                return None
            if not self.is_eligible():
                return None
            if self.initial_launch_gate_satisfied(current):
                return None
            current_count = github_star_prompt_non_negative_count(
                current_prompt.eligible_launch_count
            )
            return _with_star_state(
                current,
                eligible_launch_count=min(
                    current_count + 1,
                    GITHUB_STAR_PROMPT_ELIGIBLE_LAUNCH_THRESHOLD,
                ),
            )

        await self.persist_mutation(
            failure_context="eligible launch state",
            mutate=mutate,
        )
        settings = self.settings_provider()
        return bool(
            settings is not None
            and not _star_state(settings).clicked
            and self.is_eligible()
            and self.initial_launch_gate_satisfied(settings)
        )

    async def persist_clicked(self) -> bool:
        def mutate(settings: AppSettingsVNext) -> AppSettingsVNext | None:
            return _with_star_state(settings, clicked=True)

        return await self.persist_mutation(
            failure_context="click state",
            mutate=mutate,
        )

    async def persist_translation_success_observed(self) -> bool:
        def mutate(settings: AppSettingsVNext) -> AppSettingsVNext | None:
            if not self.has_user_owned_cloud_connection(settings):
                return None
            if _star_state(settings).translation_success_observed:
                return None
            return _with_star_state(settings, translation_success_observed=True)

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
        if _star_state(settings).translation_success_observed:
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
    ) -> object:
        await self.drain_translation_success_observation()
        async with self.persistence_lock:
            current_settings = self.settings_provider()
            if current_settings is None or not isinstance(replacement_settings, AppSettingsVNext):
                return replacement_settings
            current = _star_state(current_settings)
            replacement = _star_state(replacement_settings)
            return _with_star_state(
                replacement_settings,
                clicked=bool(replacement.clicked or current.clicked),
                translation_success_observed=bool(
                    replacement.translation_success_observed or current.translation_success_observed
                ),
                eligible_launch_count=max(
                    github_star_prompt_non_negative_count(replacement.eligible_launch_count),
                    github_star_prompt_non_negative_count(current.eligible_launch_count),
                ),
                show_count=max(
                    github_star_prompt_non_negative_count(replacement.show_count),
                    github_star_prompt_non_negative_count(current.show_count),
                ),
                last_shown_at=github_star_prompt_latest_timestamp(
                    replacement.last_shown_at,
                    current.last_shown_at,
                ),
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
        prompt = _star_state(settings)
        return (
            prompt.clicked,
            prompt.last_shown_at,
            prompt.show_count,
            prompt.translation_success_observed,
            prompt.eligible_launch_count,
        )

    @staticmethod
    def restore_state_snapshot(
        settings: GithubStarPromptSettings,
        snapshot: tuple[object, ...],
    ) -> object:
        (
            clicked,
            last_shown_at,
            show_count,
            translation_success_observed,
            eligible_launch_count,
        ) = snapshot
        if not isinstance(settings, AppSettingsVNext):
            return settings
        return _with_star_state(
            settings,
            clicked=bool(clicked),
            last_shown_at=last_shown_at if isinstance(last_shown_at, str) else None,
            show_count=github_star_prompt_non_negative_count(show_count),
            translation_success_observed=bool(translation_success_observed),
            eligible_launch_count=github_star_prompt_non_negative_count(eligible_launch_count),
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
