from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.core.stt.backend import (
    STTNativeProvenance,
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTTextContribution,
)
from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun


class STTNormalizationError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class STTNormalizationDiagnostic:
    identity: STTProviderTurnIdentity
    reason: str


class STTScopedTurnNormalizer:
    MAX_ASSEMBLY_BYTES = 1024 * 1024
    MAX_LANGUAGE_RUNS = 256
    MAX_SPEAKER_RUNS = 256

    def __init__(
        self,
        identity: STTProviderTurnIdentity,
        *,
        diagnostic_sink: Callable[[STTNormalizationDiagnostic], object] | None = None,
    ) -> None:
        self.identity = identity
        self._diagnostic_sink = diagnostic_sink
        self._stable_text = ""
        self._stable_raw_text = ""
        self._stable_raw_runs: tuple[FinalLanguageRun, ...] = ()
        self._stable_runs: tuple[FinalLanguageRun, ...] = ()
        self._stable_raw_speaker_runs: tuple[FinalSpeakerRun, ...] = ()
        self._stable_speaker_runs: tuple[FinalSpeakerRun, ...] = ()
        self._provisional_text = ""
        self._provisional_runs: tuple[FinalLanguageRun, ...] = ()
        self._provisional_speaker_runs: tuple[FinalSpeakerRun, ...] = ()
        self._last_sequence = -1
        self._native_event_ids: set[str] = set()
        self._provenance: list[STTNativeProvenance] = []
        self._contributions: list[STTTextContribution] = []
        self._terminal: STTProviderTurnTerminal | None = None

    @property
    def stable_text(self) -> str:
        return self._stable_text

    @property
    def provisional_text(self) -> str:
        return self._provisional_text

    @property
    def terminal(self) -> STTProviderTurnTerminal | None:
        return self._terminal

    def apply_update(self, update: STTProviderTurnUpdate) -> STTProviderTurnUpdate | None:
        self._require_identity(update.identity)
        if self._terminal is not None:
            return None
        native_event_id = update.provenance.native_event_id
        if native_event_id is not None and native_event_id in self._native_event_ids:
            return None
        if update.sequence <= self._last_sequence:
            raise STTNormalizationError("provider_update_sequence_disorder")
        self._last_sequence = update.sequence
        if native_event_id is not None:
            self._native_event_ids.add(native_event_id)
        self._remember_provenance(update.provenance)
        contribution: STTTextContribution | None = None
        if update.stability == "stable":
            previous_length = len(self._stable_text)
            raw_text, raw_runs, raw_speaker_runs = self._assemble(
                self._stable_raw_text,
                self._stable_raw_runs,
                self._stable_raw_speaker_runs,
                update,
            )
            text, runs = self._normalize_text_and_runs(raw_text, raw_runs)
            speaker_runs = self._normalize_speaker_runs(raw_text, raw_speaker_runs)
            if (
                update.assembly == "replace"
                and self._stable_text
                and not text.startswith(self._stable_text)
            ):
                raise STTNormalizationError("provider_stable_prefix_inconsistent")
            self._ensure_bounded(
                stable_text=text,
                stable_runs=runs,
                stable_raw_text=raw_text,
                stable_raw_runs=raw_runs,
                stable_raw_speaker_runs=raw_speaker_runs,
                stable_speaker_runs=speaker_runs,
            )
            self._stable_raw_text = raw_text
            self._stable_raw_runs = raw_runs
            self._stable_raw_speaker_runs = raw_speaker_runs
            self._stable_text = text
            self._stable_runs = runs
            self._stable_speaker_runs = speaker_runs
            text = self._stable_text
            runs = self._stable_runs
            speaker_runs = self._stable_speaker_runs
            if len(text) > previous_length:
                contribution = STTTextContribution(
                    contribution_id=f"{self.identity.provider_turn_id}:{update.sequence}",
                    text_start=previous_length,
                    text_end=len(text),
                )
                self._contributions.append(contribution)
        else:
            (
                self._provisional_text,
                self._provisional_runs,
                self._provisional_speaker_runs,
            ) = self._assemble(
                self._provisional_text,
                self._provisional_runs,
                self._provisional_speaker_runs,
                update,
            )
            text = self._provisional_text
            runs = self._provisional_runs
            speaker_runs = self._provisional_speaker_runs
        self._ensure_bounded()
        return STTProviderTurnUpdate(
            identity=update.identity,
            sequence=update.sequence,
            stability=update.stability,
            assembly="replace",
            text=text,
            final_language_runs=runs,
            final_speaker_runs=speaker_runs,
            provenance=update.provenance,
            contribution=contribution,
        )

    def apply_terminal(self, terminal: STTProviderTurnTerminal) -> STTProviderTurnTerminal:
        self._require_identity(terminal.identity)
        if self._terminal is not None:
            return self._terminal
        for item in terminal.provenance:
            if item.native_event_id is not None:
                self._native_event_ids.add(item.native_event_id)
            self._remember_provenance(item)
        raw_text = terminal.text if terminal.text else self._stable_text
        runs = terminal.final_language_runs if terminal.text else self._stable_runs
        speaker_runs = terminal.final_speaker_runs if terminal.text else self._stable_speaker_runs
        normalized_speaker_runs = self._normalize_speaker_runs(raw_text, speaker_runs)
        text, runs = self._normalize_text_and_runs(raw_text, runs)
        speaker_runs = self._normalize_speaker_runs(text, normalized_speaker_runs)
        if text and self._stable_text and not text.startswith(self._stable_text):
            raise STTNormalizationError("provider_stable_prefix_inconsistent")
        outcome = terminal.outcome
        authority = terminal.text_authority
        failure_reason = terminal.failure_reason
        if outcome == "final" and not text:
            outcome = "empty"
            authority = "authoritative"
        elif outcome == "empty" and text:
            outcome = "final"
            authority = "authoritative"
        elif outcome == "failed" and text:
            outcome = "degraded"
            authority = "degraded"
        elif outcome == "degraded" and not text:
            outcome = "failed"
            authority = "none"
        elif outcome in ("suppressed", "expired", "cancelled"):
            text = ""
            runs = ()
            speaker_runs = ()
            authority = "none"
        elif outcome == "final":
            authority = "authoritative"
        elif outcome == "empty":
            authority = "authoritative"
        elif outcome == "degraded":
            authority = "degraded"
        elif outcome == "failed":
            authority = "none"
        self._provisional_text = ""
        self._provisional_runs = ()
        self._provisional_speaker_runs = ()
        self._stable_text = text
        self._stable_runs = runs
        self._stable_speaker_runs = speaker_runs
        self._ensure_bounded()
        self._terminal = STTProviderTurnTerminal(
            identity=terminal.identity,
            outcome=outcome,
            text=text,
            final_language_runs=runs,
            final_speaker_runs=speaker_runs,
            text_authority=authority,
            failure_reason=failure_reason,
            epoch_disposition=terminal.epoch_disposition,
            provenance=tuple(self._provenance),
            included_contributions=tuple(self._contributions),
        )
        return self._terminal

    def failure_terminal(
        self,
        *,
        reason: str,
        allow_provisional: bool = False,
    ) -> STTProviderTurnTerminal:
        if self._terminal is not None:
            return self._terminal
        text = self._stable_text
        runs = self._stable_runs
        speaker_runs = self._stable_speaker_runs
        if not text and allow_provisional:
            text = self._provisional_text
            runs = self._provisional_runs
            speaker_runs = self._provisional_speaker_runs
        outcome = "degraded" if text.strip() else "failed"
        return self.apply_terminal(
            STTProviderTurnTerminal(
                identity=self.identity,
                outcome=outcome,
                text=text,
                final_language_runs=runs,
                final_speaker_runs=speaker_runs,
                text_authority="degraded" if outcome == "degraded" else "none",
                failure_reason=reason,
                epoch_disposition="retire",
                provenance=tuple(self._provenance),
            )
        )

    def _assemble(
        self,
        current_text: str,
        current_runs: tuple[FinalLanguageRun, ...],
        current_speaker_runs: tuple[FinalSpeakerRun, ...],
        update: STTProviderTurnUpdate,
    ) -> tuple[str, tuple[FinalLanguageRun, ...], tuple[FinalSpeakerRun, ...]]:
        if update.assembly == "replace":
            text = update.text
            runs = update.final_language_runs
            speaker_runs = update.final_speaker_runs
        else:
            text = current_text + update.text
            runs = current_runs + update.final_language_runs
            speaker_runs = current_speaker_runs + update.final_speaker_runs
        if not text:
            return "", (), ()
        if not runs or "".join(item.text for item in runs) != text:
            runs = self._unknown_run(text, "language_run_conservation_fallback")
        elif any(not item.language.strip() for item in runs) or len(runs) > self.MAX_LANGUAGE_RUNS:
            runs = self._unknown_run(text, "language_run_limit_fallback")
        if speaker_runs and "".join(item.text for item in speaker_runs) != text:
            self._diagnose("speaker_run_conservation_fallback")
            speaker_runs = ()
        return text, runs, speaker_runs

    def _normalize_text_and_runs(
        self,
        text: str,
        runs: tuple[FinalLanguageRun, ...],
    ) -> tuple[str, tuple[FinalLanguageRun, ...]]:
        normalized = text.strip()
        if not normalized:
            return "", ()
        if not runs or "".join(item.text for item in runs) != text:
            return normalized, self._unknown_run(normalized, "language_run_conservation_fallback")
        trimmed = list(runs)
        left = len(text) - len(text.lstrip())
        right = len(text) - len(text.rstrip())
        while left and trimmed:
            item = trimmed[0]
            amount = min(left, len(item.text))
            item = FinalLanguageRun(item.text[amount:], item.language)
            left -= amount
            if item.text:
                trimmed[0] = item
            else:
                trimmed.pop(0)
        while right and trimmed:
            item = trimmed[-1]
            amount = min(right, len(item.text))
            item = FinalLanguageRun(item.text[: len(item.text) - amount], item.language)
            right -= amount
            if item.text:
                trimmed[-1] = item
            else:
                trimmed.pop()
        if any(not item.language.strip() for item in trimmed):
            return normalized, self._unknown_run(normalized, "invalid_language_run_fallback")
        merged: list[FinalLanguageRun] = []
        for item in trimmed:
            if not item.text:
                continue
            if merged and merged[-1].language == item.language:
                previous = merged[-1]
                merged[-1] = FinalLanguageRun(previous.text + item.text, item.language)
            else:
                merged.append(item)
        if (
            len(merged) > self.MAX_LANGUAGE_RUNS
            or "".join(item.text for item in merged) != normalized
        ):
            return normalized, self._unknown_run(normalized, "language_run_limit_fallback")
        return normalized, tuple(merged)

    def _normalize_speaker_runs(
        self,
        text: str,
        runs: tuple[FinalSpeakerRun, ...],
    ) -> tuple[FinalSpeakerRun, ...]:
        if not runs:
            return ()
        if "".join(item.text for item in runs) != text:
            self._diagnose("speaker_run_conservation_fallback")
            return ()
        if len(runs) > self.MAX_SPEAKER_RUNS:
            self._diagnose("speaker_run_limit_fallback")
            return ()
        left = len(text) - len(text.lstrip())
        right = len(text) - len(text.rstrip())
        trimmed = list(runs)
        while left and trimmed:
            item = trimmed[0]
            amount = min(left, len(item.text))
            item = FinalSpeakerRun(
                item.text[amount:],
                item.speaker_id,
                item.session_scope,
            )
            left -= amount
            if item.text:
                trimmed[0] = item
            else:
                trimmed.pop(0)
        while right and trimmed:
            item = trimmed[-1]
            amount = min(right, len(item.text))
            item = FinalSpeakerRun(
                item.text[: len(item.text) - amount],
                item.speaker_id,
                item.session_scope,
            )
            right -= amount
            if item.text:
                trimmed[-1] = item
            else:
                trimmed.pop()
        merged: list[FinalSpeakerRun] = []
        for item in trimmed:
            if not item.text or not item.session_scope.strip():
                continue
            if (
                merged
                and merged[-1].speaker_id == item.speaker_id
                and merged[-1].session_scope == item.session_scope
            ):
                previous = merged[-1]
                merged[-1] = FinalSpeakerRun(
                    previous.text + item.text,
                    item.speaker_id,
                    item.session_scope,
                )
            else:
                merged.append(item)
        normalized = text.strip()
        if "".join(item.text for item in merged) != normalized:
            self._diagnose("invalid_speaker_run_fallback")
            return ()
        return tuple(merged)

    def _unknown_run(self, text: str, reason: str) -> tuple[FinalLanguageRun, ...]:
        self._diagnose(reason)
        return (FinalLanguageRun(text=text, language="unknown"),)

    def _ensure_bounded(
        self,
        *,
        stable_text: str | None = None,
        stable_runs: tuple[FinalLanguageRun, ...] | None = None,
        stable_raw_text: str | None = None,
        stable_raw_runs: tuple[FinalLanguageRun, ...] | None = None,
        stable_speaker_runs: tuple[FinalSpeakerRun, ...] | None = None,
        stable_raw_speaker_runs: tuple[FinalSpeakerRun, ...] | None = None,
    ) -> None:
        bounded_stable_text = self._stable_text if stable_text is None else stable_text
        bounded_stable_runs = self._stable_runs if stable_runs is None else stable_runs
        bounded_raw_text = self._stable_raw_text if stable_raw_text is None else stable_raw_text
        bounded_raw_runs = self._stable_raw_runs if stable_raw_runs is None else stable_raw_runs
        bounded_speaker_runs = (
            self._stable_speaker_runs if stable_speaker_runs is None else stable_speaker_runs
        )
        bounded_raw_speaker_runs = (
            self._stable_raw_speaker_runs
            if stable_raw_speaker_runs is None
            else stable_raw_speaker_runs
        )
        size = sum(
            len(text.encode("utf-8"))
            for text in (
                bounded_stable_text,
                bounded_raw_text,
                self._provisional_text,
            )
        )
        for run in bounded_stable_runs + self._provisional_runs:
            size += len(run.language.encode("utf-8"))
        for run in bounded_raw_runs:
            size += len(run.text.encode("utf-8")) + len(run.language.encode("utf-8"))
        for run in bounded_speaker_runs + self._provisional_speaker_runs:
            size += len(run.text.encode("utf-8")) + len(run.session_scope.encode("utf-8"))
            if run.speaker_id is not None:
                size += len(run.speaker_id.encode("utf-8"))
        for run in bounded_raw_speaker_runs:
            size += len(run.text.encode("utf-8")) + len(run.session_scope.encode("utf-8"))
            if run.speaker_id is not None:
                size += len(run.speaker_id.encode("utf-8"))
        for provenance in self._provenance:
            size += sum(
                len(value.encode("utf-8"))
                for value in (
                    provenance.native_event_id,
                    provenance.native_request_id,
                    provenance.native_item_id,
                    provenance.native_task_id,
                    provenance.barrier,
                )
                if value is not None
            )
        if size > self.MAX_ASSEMBLY_BYTES:
            raise STTNormalizationError("provider_result_too_large")

    def _remember_provenance(self, provenance: STTNativeProvenance) -> None:
        if (
            provenance.native_event_id is None
            and provenance.native_request_id is None
            and provenance.native_item_id is None
            and provenance.native_task_id is None
            and provenance.barrier is None
            and provenance.from_finalize is None
        ):
            return
        if provenance not in self._provenance:
            self._provenance.append(provenance)

    def _require_identity(self, identity: STTProviderTurnIdentity) -> None:
        if identity != self.identity:
            raise STTNormalizationError("provider_turn_identity_mismatch")

    def _diagnose(self, reason: str) -> None:
        if self._diagnostic_sink is not None:
            self._diagnostic_sink(STTNormalizationDiagnostic(identity=self.identity, reason=reason))


__all__ = [
    "STTNormalizationDiagnostic",
    "STTNormalizationError",
    "STTScopedTurnNormalizer",
]
