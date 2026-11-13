from __future__ import annotations

import asyncio
import json
import re
import secrets
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from time import monotonic

from puripuly_heart import __version__
from puripuly_heart.app.ports.agent_chat import (
    AgentChatSnapshot,
    AgentContextMode,
    AgentDecisionEntry,
    AgentEffort,
    AgentSecureEntryPort,
    AgentSecureEntryRequest,
    AgentServiceError,
    AgentSetupAction,
    AgentToolResult,
    AgentTransport,
    AgentTransportError,
    AgentTurnObservation,
    AgentTurnResult,
)
from puripuly_heart.app.ports.application_control import ApplicationControl
from puripuly_heart.app.services.agent_chat.context import AgentActivity, collect_agent_context
from puripuly_heart.app.services.agent_chat.conversation import AgentConversation
from puripuly_heart.app.services.agent_chat.execution import (
    APP_TOOLS,
    select_app_action,
    validate,
)
from puripuly_heart.app.services.agent_chat.modules import (
    MODULES,
    confirmation_modules,
    model_context,
    model_receipt,
    ordered_modules,
    provider_modules,
)
from puripuly_heart.app.services.agent_chat.presentation import AgentPresentationBuffer
from puripuly_heart.app.services.agent_chat.provider_setup import (
    ACCOUNT_SETUP_ACTIONS,
    PROVIDER_SETUPS,
)
from puripuly_heart.app.services.agent_chat.report import (
    MAX_SUPPORT_BYTES,
    SupportReport,
    build_support_bundle,
    recent_diagnostics,
)
from puripuly_heart.app.services.agent_chat.routing import (
    AgentDecision,
    classification_input,
    extract_decision,
)
from puripuly_heart.app.services.agent_chat.tools import AgentTools, project_operation_receipt
from puripuly_heart.core.diagnostic_validation import contains_known_secret_pattern
from puripuly_heart.core.diagnostics import (
    exception_diagnostics,
    normalize_diagnostic,
    serialize_diagnostics,
)
from puripuly_heart.core.lifecycle import LifecycleScope
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.owned_thread import run_owned_thread_call


@dataclass(frozen=True, slots=True)
class _ContextScope:
    modules: tuple[str, ...] = ()
    message_index: int | None = None
    summary: str | None = None


@dataclass(frozen=True, slots=True)
class _AppNotice:
    item: dict
    index: int
    effort: AgentEffort
    context: _ContextScope


class AgentChatOwner:
    def __init__(
        self,
        control: ApplicationControl,
        transport: AgentTransport | None,
        *,
        localize: Callable[[str], str],
        support_log_snapshot: Callable[[], bytes] | None = None,
        decision_timeout: float = 1.0,
        context_mode: AgentContextMode = "selected",
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._control = control
        self._transport = transport
        self._localize = localize
        self._decision_timeout = decision_timeout
        if context_mode not in ("selected", "full"):
            raise ValueError("unsupported context mode")
        self._context_mode = context_mode
        self._clock = clock
        self._support_log_snapshot = support_log_snapshot
        self._reports: dict[str, SupportReport] = {}
        self._report_tasks: dict[str, asyncio.Task] = {}
        self._last_activity: float | None = None
        self._last_effort: AgentEffort = "high"
        self._context_scope = _ContextScope()
        self._card_context: tuple[str, _ContextScope] | None = None
        self._watched_context = _ContextScope()
        self._activity = AgentActivity()
        self._scope = LifecycleScope("agent-chat")
        self._started = False
        self._closed = False
        self._generation = 0
        self._notice_epoch = 0
        self._active: asyncio.Task | None = None
        self._notice_task: asyncio.Task | None = None
        self._conversation = AgentConversation()
        self._presentation: AgentPresentationBuffer | None = None
        self._items = self._conversation.items
        self._messages = self._conversation.messages
        self._results: list[AgentToolResult] = []
        self._pending_user: str | None = None
        self._prompt_turn_id: str | None = None
        self._prompt_requests = 0
        self._prompt_started = 0.0
        self._actions: deque[dict] = deque(maxlen=5)
        self._operations: dict[str, dict] = {}
        self._pending_notices: deque[_AppNotice] = deque()
        self._quiet_notices: deque[tuple[int, dict]] = deque()
        self._active_tool: tuple[str, str | None] | None = None
        self._verification_providers: frozenset[str] = frozenset()
        self._watched_provider: str | None = None
        self._confirming: dict[str, tuple[int, _ContextScope]] = {}
        self._allowance_exhausted = False
        self._observations: deque[AgentTurnObservation] = deque(maxlen=20)
        self._observers: set[Callable[[AgentChatSnapshot], None]] = set()
        self._setup_actions: dict[str, AgentSetupAction] = {}
        self._secure_requests: dict[str, AgentSecureEntryRequest] = {}
        self._setup_action_handler: Callable[[str], dict] | None = None
        self._secure_entry: AgentSecureEntryPort | None = None
        self._phase = "idle"
        self._event_observers: set[Callable[[dict], bool]] = set()
        self._tools = AgentTools(
            control,
            changed=lambda: self._publish(self._phase),
            record_action=self._record_action,
            emit_event=self._tool_event,
        )
        self._tools.bind_localize(localize)

    @property
    def snapshot(self) -> AgentChatSnapshot:
        return AgentChatSnapshot(
            enabled=self._transport is not None,
            busy=self._active is not None,
            phase=self._phase,
            messages=tuple(self._messages),
            tool_results=tuple(deepcopy(self._results)),
            confirmation=self._tools.confirmation,
            recent_changes=tuple(deepcopy(self._actions)),
            report_offers=tuple(report.snapshot for report in self._reports.values()),
            presentation=self._presentation.snapshot if self._presentation is not None else None,
            setup_actions=tuple(self._setup_actions.values()),
            secure_entry_requests=tuple(self._secure_requests.values()),
            confirmation_previous=(
                (self._tools.confirmation_display or {}).get("previous")
                if self._tools.confirmation is not None
                else None
            ),
            active_tool=self._active_tool,
        )

    @property
    def observations(self) -> tuple[AgentTurnObservation, ...]:
        return tuple(self._observations)

    async def open_secure_settings(self) -> dict:
        if self._closed:
            return {"status": "rejected", "code": "closed"}
        return await self._tools.execute(
            "open_app", {"view": "settings", "section": "api"}, require_current=lambda: None
        )

    @property
    def report_archive(self) -> tuple[dict, ...]:
        return tuple(deepcopy(self._conversation.archive))

    def archive_support_report(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        self._conversation.display("notice", text, grouped=False)
        self._conversation.record({"type": "support_report", "role": "system", "content": text})
        self._quiet_notices.append((len(self._messages), {"role": "user", "content": text}))
        self._publish(self._phase)

    def _propose_report(self, summary: str, generation: int) -> dict:
        if not summary.strip():
            return {"status": "rejected", "code": "invalid_tool_call"}
        for report in self._reports.values():
            if report.generation == generation:
                return {"status": report.status, "offer_id": report.id}
        identity = f"support-{len(self._reports) + 1}"
        self._reports[identity] = SupportReport(identity, generation, summary)
        return {"status": "pending", "offer_id": identity}

    def mark_report_displayed(self, offer_id: str) -> dict:
        report = self._reports.get(offer_id)
        if self._closed or report is None or report.message_index is None:
            return {"status": "rejected", "code": "unavailable_offer"}
        report.displayed = True
        return {"status": "displayed", "offer_id": offer_id}

    def reject_report(self, offer_id: str) -> dict:
        report = self._reports.get(offer_id)
        if self._closed or report is None or report.status not in {"pending", "offered"}:
            return {"status": "rejected", "code": "unavailable_offer"}
        report.status = "rejected"
        self._record_quiet_notice(
            f"[app notice] support_report offer_id={offer_id} result=declined"
        )
        self._publish(self._phase)
        return {"status": "rejected", "offer_id": offer_id}

    async def send_report(self, offer_id: str) -> dict:
        if self._closed or not self._started or self._transport is None:
            return {"status": "rejected", "code": "closed"}
        report = self._reports.get(offer_id) if isinstance(offer_id, str) else None
        if report is None:
            return {"status": "rejected", "code": "unknown_offer"}
        if report.status in {"sending", "sent"}:
            return {"status": report.status, "ticket": report.ticket}
        if report.message_index is None or report.status not in {"offered", "failed"}:
            return {"status": "rejected", "code": "unavailable_offer"}
        report.clicked = True
        if report.ticket is None:
            report.ticket = "PP-" + secrets.token_hex(8).upper()
            report.archive = self.report_archive
        report.status = "sending"
        report.attempt += 1
        task = self._scope.create_task(
            self._send_report(report), name=f"{report.id}-{report.attempt}"
        )
        self._report_tasks[report.id] = task
        self._publish(self._phase)
        return await asyncio.shield(task)

    async def _send_report(self, report: SupportReport) -> dict:
        stage = "bundle"
        try:
            ticket = report.ticket
            if ticket is None or self._transport is None:
                raise RuntimeError("support_report_not_admitted")
            if report.bundle is None:
                log_snapshot = self._support_log_snapshot
                archive = report.archive
                if archive is None:
                    raise RuntimeError("support_archive_unavailable")
                if report.diagnostics is None:
                    report.diagnostics = await recent_diagnostics(self._control)
                diagnostics = report.diagnostics
                report.bundle = await run_owned_thread_call(
                    lambda: build_support_bundle(
                        ticket=ticket,
                        app_version=__version__,
                        summary=report.summary,
                        diagnostics=diagnostics,
                        archive=archive,
                        log_snapshot=log_snapshot,
                    )
                )
                report.archive = None
            if len(report.bundle) > MAX_SUPPORT_BYTES:
                raise ValueError("support_bundle_too_large")
            stage = "upload"
            receipt = await self._transport.upload_support(
                bundle=report.bundle,
                ticket=ticket,
                app_version=__version__,
                diagnostic_codes=report.diagnostic_codes or (),
            )
            if not isinstance(receipt, dict) or receipt.get("ticket") != report.ticket:
                raise AgentTransportError("invalid_support_receipt")
            report.status = "sent"
            report.bundle = None
            self.archive_support_report(f"[app notice] support_report sent ticket={report.ticket}")
            return {"status": "sent", "ticket": report.ticket}
        except asyncio.CancelledError:
            report.status = "failed"
            raise
        except Exception as exc:
            report.status = "failed"
            return {
                "status": "failed",
                "ticket": report.ticket,
                "diagnostics": self._failure_evidence(
                    exc, code="support_report_failed", operation="support_report", stage=stage
                ),
            }
        finally:
            self._report_tasks.pop(report.id, None)
            self._publish(self._phase)

    def _failure_evidence(
        self, exception: Exception, *, code: str, operation: str, stage: str
    ) -> dict:
        return normalize_diagnostic(
            self._failure(exception, code=code, operation=operation),
            instance_id=getattr(self._control, "instance_id", None),
            severity="error",
            disposition="failed",
            stage=stage,
            file_evidence="unavailable",
        )

    def _failure(self, exception: Exception, *, code: str, operation: str) -> ErrorDiagnostics:
        return exception_diagnostics(
            exception,
            component="application_control",
            operation=operation,
            code=code,
            category="unknown",
            fields=(
                {"service_code": exception.code}
                if isinstance(exception, AgentServiceError)
                else None
            ),
        )

    def observe(self, callback: Callable[[AgentChatSnapshot], None]) -> Callable[[], None]:
        self._observers.add(callback)
        return lambda: self._observers.discard(callback)

    def bind_navigation(self, callback: Callable[[str, str | None], dict] | None) -> None:
        self._tools.bind_navigation(callback)

    def bind_verification_providers(self, providers: frozenset[str]) -> None:
        self._verification_providers = frozenset(providers)
        if self._watched_provider not in self._verification_providers:
            self._watched_provider = None

    def bind_setup_actions(self, callback: Callable[[str], dict] | None) -> None:
        self._setup_action_handler = callback
        for identity, action in tuple(self._setup_actions.items()):
            if action.generation == self._generation and action.message_index is not None:
                self._setup_actions[identity] = replace(
                    action, available=callback is not None and action.status != "cancelled"
                )

    def bind_secure_entry(self, controller: AgentSecureEntryPort | None) -> None:
        if self._secure_entry is not None:
            self._secure_entry.cancel()
        self._secure_entry = controller
        for identity, request in tuple(self._secure_requests.items()):
            if (
                request.generation == self._generation
                and request.message_index is not None
                and request.status not in {"cancelled", "verified"}
            ):
                self._secure_requests[identity] = replace(
                    request,
                    available=controller is not None,
                    status="ready" if request.status == "verifying" else request.status,
                )

    def _stage_provider_setup(self, provider: str, generation: int) -> None:
        setup = PROVIDER_SETUPS.get(provider)
        if setup is None:
            return
        task_id = self._prompt_turn_id
        if task_id is None:
            return
        identity = f"secure-{generation}-{provider}"
        for request in tuple(self._secure_requests.values()):
            if request.generation == generation and request.provider != provider:
                self._cancel_secure_request(request.id)
        if identity not in self._secure_requests:
            self._secure_requests[identity] = AgentSecureEntryRequest(
                identity, provider, task_id, generation
            )
        for link in setup.links:
            action_id = link.action_id(provider)
            identity = f"{generation}:{action_id}"
            self._setup_actions.setdefault(
                identity,
                AgentSetupAction(
                    identity,
                    action_id,
                    provider,
                    link.kind.value,
                    generation,
                    secure_request_id=f"secure-{generation}-{provider}",
                ),
            )

    def _stage_account_actions(self, context: dict, generation: int) -> bool:
        accounts = context.get("accounts")
        if not isinstance(accounts, dict):
            return False
        staged = False
        for action in ACCOUNT_SETUP_ACTIONS.values():
            identity = f"{generation}:{action.id}"
            if identity not in self._setup_actions:
                self._setup_actions[identity] = AgentSetupAction(
                    identity, action.id, action.provider, action.kind, generation
                )
                staged = True
        return staged

    def _current_setup_action(self, identity: str) -> AgentSetupAction | None:
        action = self._setup_actions.get(identity) if isinstance(identity, str) else None
        return (
            action
            if not self._closed
            and action is not None
            and action.generation == self._generation
            and action.message_index is not None
            and action.available
            else None
        )

    def mark_setup_action_displayed(self, action_id: str) -> dict:
        action = self._current_setup_action(action_id)
        if action is None:
            return {"status": "rejected", "code": "stale_action"}
        self._setup_actions[action.id] = replace(action, displayed=True)
        return {"status": "displayed", "action_id": action.id}

    def activate_setup_action(self, action_id: str) -> dict:
        action = self._current_setup_action(action_id)
        if action is None:
            return {"status": "rejected", "code": "stale_action"}
        if not action.displayed:
            return {"status": "rejected", "code": "action_not_displayed"}
        handler = self._setup_action_handler
        if handler is None:
            return {"status": "unavailable", "code": "action_handler_unavailable"}
        self._setup_actions[action.id] = replace(action, clicked=True)
        try:
            result = handler(action.action_id)
        except Exception:
            result = {"status": "failed"}
        status = result.get("status") if isinstance(result, dict) else None
        if status not in {"opened", "failed", "unavailable"}:
            status = "failed"
        current = self._setup_actions[action.id]
        self._setup_actions[action.id] = replace(current, status=status)
        self._publish(self._phase)
        return {"status": status, "action_id": action.id, "provider": action.provider}

    def _current_secure_request(self, identity: str) -> AgentSecureEntryRequest | None:
        request = self._secure_requests.get(identity) if isinstance(identity, str) else None
        return (
            request
            if not self._closed
            and request is not None
            and request.generation == self._generation
            and request.task_id == self._prompt_turn_id
            and request.available
            and request.status not in {"cancelled", "verified"}
            else None
        )

    def mark_secure_entry_displayed(self, request_id: str) -> dict:
        request = self._current_secure_request(request_id)
        if request is None:
            return {"status": "rejected", "code": "stale_request"}
        self._secure_requests[request.id] = replace(request, displayed=True)
        return {"status": "displayed", "request_id": request.id}

    async def submit_secure_entry(self, request_id: str, value: str) -> dict:
        request = self._current_secure_request(request_id)
        if request is None:
            return {"status": "rejected", "code": "stale_request"}
        if self._secure_entry is None:
            return {"status": "unavailable", "code": "secure_entry_unavailable"}
        if not isinstance(value, str) or not value.strip():
            return {"status": "rejected", "code": "empty_credential"}
        if request.status == "verifying":
            return {"status": "rejected", "code": "verification_in_progress"}
        controller = self._secure_entry
        self._secure_requests[request.id] = replace(request, status="verifying", code=None)
        self._watched_provider = request.provider
        self._watched_context = self._context_scope
        completed = False
        failure_diagnostics = None

        def complete(
            provider: str,
            success: bool,
            code: str | None,
            diagnostics: ErrorDiagnostics | None = None,
        ) -> None:
            nonlocal completed, failure_diagnostics
            current = self._current_secure_request(request.id)
            if (
                completed
                or current is None
                or self._secure_entry is not controller
                or provider != request.provider
            ):
                return
            completed = True
            safe_code = (
                code
                if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_.]{0,63}", code)
                else None
            )
            self._secure_requests[request.id] = replace(
                current,
                status="verified" if success else "failed",
                available=False if success else current.available,
                code=None if success else safe_code or "verification_failed",
            )
            failure_diagnostics = diagnostics if not success else None
            self.key_verification_completed(provider, success, safe_code, diagnostics)

        self._publish(self._phase)
        try:
            result = await controller.submit(
                request,
                value,
                is_current=lambda: (
                    self._secure_entry is controller
                    and self._current_secure_request(request.id) is not None
                ),
                complete=complete,
            )
        except asyncio.CancelledError:
            self._cancel_secure_request(request.id)
            raise
        if (
            result.get("status") == "cancelled"
            and self._current_secure_request(request.id) is not None
        ):
            self._cancel_secure_request(request.id)
        self._publish(self._phase)
        if isinstance(failure_diagnostics, ErrorDiagnostics):
            result = {**result, "diagnostics": serialize_diagnostics(failure_diagnostics)}
        return result

    def cancel_secure_entry(self, request_id: str) -> dict:
        result = self._cancel_secure_request(request_id)
        if result["status"] == "cancelled":
            provider = self._secure_requests[request_id].provider
            self._record_quiet_notice(
                f"[app notice] secure_entry provider={provider} result=cancelled"
            )
        return result

    def _cancel_secure_request(self, request_id: str) -> dict:
        request = self._secure_requests.get(request_id) if isinstance(request_id, str) else None
        if (
            self._closed
            or request is None
            or request.generation != self._generation
            or request.task_id != self._prompt_turn_id
            or request.status in {"verified", "cancelled"}
        ):
            return {"status": "rejected", "code": "stale_request"}
        self._secure_requests[request.id] = replace(
            request, available=False, status="cancelled", code=None
        )
        for identity, action in tuple(self._setup_actions.items()):
            if action.secure_request_id == request.id:
                self._setup_actions[identity] = replace(action, available=False, status="cancelled")
        if self._secure_entry is not None:
            self._secure_entry.cancel()
        if self._watched_provider == request.provider:
            self._watched_provider = None
        self._publish(self._phase)
        return {"status": "cancelled", "request_id": request.id}

    async def open_secure_entry_settings(self, request_id: str) -> dict:
        request = self._secure_requests.get(request_id) if isinstance(request_id, str) else None
        if (
            self._closed
            or request is None
            or request.generation != self._generation
            or request.task_id != self._prompt_turn_id
            or request.message_index is None
            or request.status in {"cancelled", "verified"}
        ):
            return {"status": "rejected", "code": "stale_request"}
        if self._secure_entry is not None:
            self._secure_entry.cancel()
        self._secure_requests[request.id] = replace(request, status="ready", code=None)
        generation = self._generation
        return await self._tools.execute(
            "open_app",
            {"view": "settings", "section": "api", "provider": request.provider},
            require_current=lambda: self._require_current(generation),
        )

    def key_verification_completed(
        self,
        provider: str,
        success: bool,
        code: str | None = None,
        diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        if (
            self._closed
            or not self._started
            or type(success) is not bool
            or provider != self._watched_provider
            or (
                provider not in self._verification_providers
                and not any(
                    request.provider == provider and request.generation == self._generation
                    for request in self._secure_requests.values()
                )
            )
        ):
            return
        safe_code = (
            code if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_.]{0,63}", code) else None
        )
        if success:
            self._watched_provider = None
        else:
            safe_code = safe_code or "verification_failed"
        for identity, request in tuple(self._secure_requests.items()):
            if request.provider == provider and request.generation == self._generation:
                self._secure_requests[identity] = replace(
                    request,
                    status="verified" if success else "failed",
                    available=False if success else request.available,
                    code=None if success else safe_code,
                )
        text = f"[app notice] key_verification provider={provider} result={'success' if success else 'failed'}"
        if safe_code is not None:
            text += f" code={safe_code}"
        if isinstance(diagnostics, ErrorDiagnostics):
            text += " diagnostics=" + json.dumps(
                serialize_diagnostics(diagnostics), separators=(",", ":")
            )
        if success:
            self._accept_notice(
                text,
                replace(
                    self._watched_context,
                    modules=ordered_modules(
                        (*self._watched_context.modules, *provider_modules(provider))
                    ),
                ),
            )
        else:
            self._record_quiet_notice(text)

    def observe_events(self, callback: Callable[[dict], bool]) -> Callable[[], None]:
        self._event_observers.add(callback)
        return lambda: self._event_observers.discard(callback)

    def _emit_event(self, event: dict) -> None:
        if not self._closed:
            for callback in tuple(self._event_observers):
                try:
                    callback(deepcopy(event))
                except Exception:
                    self._event_observers.discard(callback)

    def _tool_event(self, event: dict) -> bool:
        if self._closed:
            return False
        observable = False
        if event.get("kind") == "navigation_requested":
            provider = event.get("provider")
            if event.get("section") == "api" and provider is not None:
                self._watched_provider = (
                    provider if provider in self._verification_providers else None
                )
                observable = self._watched_provider is not None
                if observable:
                    self._watched_context = self._context_scope
        elif event.get("kind") == "confirmation":
            card_id = event.get("card_id")
            confirmed = self._confirming.pop(card_id, None)
            if confirmed is not None and confirmed[0] == self._notice_epoch:
                context = confirmed[1]
                next_card = event["result"].get("confirmation")
                self._card_context = (
                    (
                        next_card["id"],
                        replace(
                            context,
                            modules=ordered_modules(
                                (*context.modules, *confirmation_modules(next_card))
                            ),
                        ),
                    )
                    if isinstance(next_card, dict)
                    else None
                )
                text = "[app notice] confirmation " + json.dumps(
                    {"card_id": card_id, "result": event["result"]}, ensure_ascii=False
                )
                if event["result"].get("status") == "declined" or not event.get("resume", True):
                    self._record_quiet_notice(text)
                else:
                    self._accept_notice(text, context)
        self._emit_event(event)
        return observable

    def _accept_notice(self, text: str, context: _ContextScope) -> None:
        self._last_activity = self._clock()
        item = {"role": "user", "content": text}
        index = self._conversation.display("notice", text, grouped=False)
        self._conversation.record(item)
        self._pending_notices.append(_AppNotice(item, index, self._last_effort, context))
        self._emit_event({"kind": "app_notice", "text": text})
        self._publish(self._phase)
        self._schedule_notices()

    def _record_quiet_notice(self, text: str) -> None:
        self._last_activity = self._clock()
        item = {"role": "user", "content": text}
        self._conversation.record(item)
        self._quiet_notices.append((len(self._messages), item))

    def _append_quiet_notices(self, before: int | None = None) -> None:
        while self._quiet_notices and (before is None or self._quiet_notices[0][0] <= before):
            self._conversation.append(self._quiet_notices.popleft()[1], archive=False)

    def _append_notice(self, notice: _AppNotice) -> None:
        self._append_quiet_notices(notice.index)
        self._conversation.append(notice.item, archive=False)
        self._conversation.attach(notice.index)

    def _schedule_notices(self) -> None:
        if (
            self._closed
            or self._prompt_turn_id is None
            or self._active is not None
            or not self._pending_notices
            or self._transport is None
            or self._notice_task is not None
        ):
            return
        self._notice_task = self._scope.create_task(
            self._drain_notices(), name=f"notices-{self._generation}"
        )

    async def _drain_notices(self) -> None:
        try:
            await asyncio.sleep(0)
            while (
                not self._closed
                and self._active is None
                and self._pending_notices
                and self._prompt_turn_id is not None
            ):
                notice = self._pending_notices.popleft()
                task = self._begin_turn(
                    "", entry="notice", turn_id=self._prompt_turn_id, notice=notice
                )
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if self._closed:
                        raise
                    return
        finally:
            self._notice_task = None
            self._schedule_notices()

    async def confirm(self, card_id: str, approved: bool) -> dict:
        if self._closed:
            return {"status": "rejected", "code": "closed"}
        card = self._tools.confirmation
        if type(approved) is bool and card is not None and card["id"] == card_id:
            context = (
                self._card_context[1]
                if self._card_context is not None and self._card_context[0] == card_id
                else _ContextScope(MODULES)
            )
            self._confirming[card_id] = (
                self._notice_epoch,
                replace(
                    context,
                    modules=ordered_modules((*context.modules, *confirmation_modules(card))),
                ),
            )
        epoch = self._notice_epoch
        return await self._tools.confirm(
            card_id, approved, require_current=lambda: self._require_notice_epoch(epoch)
        )

    def cancel_confirmation(self) -> None:
        self._confirming.clear()
        self._card_context = None
        self._tools.cancel_confirmation()

    def _publish(self, phase: str) -> None:
        self._phase = phase
        if phase != "working":
            self._active_tool = None
        for callback in tuple(self._observers):
            try:
                callback(self.snapshot)
            except Exception:
                self._observers.discard(callback)

    async def start(self) -> None:
        if self._closed or self._started:
            return
        self._started = True
        self._scope.create_task(self._listen(), name="activity", eager_start=True)

    async def _listen(self) -> None:
        self._activity.availability = "subscribed"
        try:
            async for event in self._control.subscribe(
                topics=["transcript", "translation", "capture"],
                include_transcripts=False,
                include_translations=False,
            ):
                if self._closed:
                    return
                self._activity.record(event)
        except Exception:
            self._activity.clear()
            self._activity.availability = "unavailable"
        finally:
            if self._closed:
                self._activity.availability = "closed"

    def _interrupt(self) -> None:
        for report in self._reports.values():
            if report.status == "pending":
                report.status = "cancelled"
        if self._secure_entry is not None:
            self._secure_entry.cancel()
        for identity, request in tuple(self._secure_requests.items()):
            if request.status not in {"verified", "cancelled"}:
                self._secure_requests[identity] = replace(
                    request, available=False, status="cancelled", code=None
                )
        for identity, action in tuple(self._setup_actions.items()):
            self._setup_actions[identity] = replace(action, available=False)
        self._generation += 1
        active, self._active = self._active, None
        self._presentation = None
        if self._pending_user is not None:
            self._append_user(self._pending_user)
        self._close_outstanding_calls()
        if active is not None:
            active.cancel()

    def cancel(self) -> None:
        if self._closed:
            return
        self._interrupt()
        self._notice_epoch += 1
        self._watched_provider = None
        self._confirming.clear()
        self._card_context = None
        self._watched_context = _ContextScope()
        self._context_scope = _ContextScope()
        while self._pending_notices:
            self._append_notice(self._pending_notices.popleft())
        self._tools.cancel_confirmation()
        self._finish("cancelled", "Puri stopped its next actions. Already applied settings remain.")
        self._publish("idle")

    def new_conversation(self) -> None:
        if not self._closed:
            self._reset_conversation("새 대화로 시작했어요")

    def _reset_conversation(self, separator: str) -> None:
        self._interrupt()
        self._notice_epoch += 1
        self._watched_provider = None
        self._confirming.clear()
        self._pending_notices.clear()
        self._quiet_notices.clear()
        self._tools.cancel_confirmation()
        self._tools.reset_context()
        self._conversation.reset(separator)
        self._context_scope = _ContextScope()
        self._card_context = None
        self._watched_context = _ContextScope()
        self._prompt_turn_id = None
        self._prompt_requests = 0
        self._prompt_started = 0.0
        self._last_effort = "high"
        self._last_activity = None
        self._results = []
        self._publish("idle")

    def freeze_ingress(self) -> None:
        self._closed = True
        self._interrupt()
        self._notice_epoch += 1
        self._watched_provider = None
        self._confirming.clear()
        self._pending_notices.clear()
        self._quiet_notices.clear()
        self._tools.cancel_confirmation()

    async def close(self) -> None:
        self.freeze_ingress()
        if self._report_tasks:
            await asyncio.shield(
                asyncio.gather(*tuple(self._report_tasks.values()), return_exceptions=True)
            )
        await self._scope.close()
        await self._tools.close()
        if self._transport is not None:
            await self._transport.close()
        self._conversation.clear_model_context()
        self._observers.clear()
        self._event_observers.clear()
        self._phase = "closed"

    async def send(self, text: str, *, diagnostic_id: str | None = None) -> AgentTurnResult:
        if self._closed or not self._started or self._transport is None:
            return AgentTurnResult("unavailable", "Agent Chat is not enabled.")
        if not isinstance(text, str) or not text.strip():
            return AgentTurnResult("rejected", "Enter a message first.")
        if text.strip().casefold() in {"/cancel", "cancel", "stop", "취소", "그만", "중지"}:
            self.cancel()
            return AgentTurnResult("cancelled", self._messages[-1].text, tuple(self._results))
        if contains_known_secret_pattern(text):
            result = self._finish("rejected", self._localize("agent_chat.key_blocked"))
            self._publish(self._phase)
            return result
        self._allowance_exhausted = False
        if (
            self._active is None
            and self._last_activity is not None
            and self._clock() - self._last_activity >= 1800
        ):
            self._reset_conversation("30분이 지나 새 대화로 시작했어요")
        self._interrupt()
        self._last_activity = self._clock()
        self._prompt_turn_id = secrets.token_hex(16)
        self._prompt_requests = 0
        self._prompt_started = self._clock()
        task = self._begin_turn(
            text,
            entry="error" if diagnostic_id is not None else "message",
            turn_id=self._prompt_turn_id,
            diagnostic_id=diagnostic_id,
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if self._active is task:
                self._interrupt()
                self._publish("idle")
            raise

    def _begin_turn(
        self,
        text: str,
        *,
        entry: AgentDecisionEntry,
        turn_id: str,
        diagnostic_id: str | None = None,
        notice: _AppNotice | None = None,
    ) -> asyncio.Task:
        providers = {
            request.provider
            for request in self._secure_requests.values()
            if entry == "notice"
            and request.generation == self._generation
            and request.message_index is not None
            and request.status in {"ready", "failed", "verifying"}
        }
        if entry == "notice":
            if self._secure_entry is not None:
                self._secure_entry.cancel()
            for identity, request in tuple(self._secure_requests.items()):
                self._secure_requests[identity] = replace(request, available=False)
            for identity, action in tuple(self._setup_actions.items()):
                self._setup_actions[identity] = replace(action, available=False)
        self._generation += 1
        generation = self._generation
        for provider in providers:
            self._stage_provider_setup(provider, generation)
        results: list[AgentToolResult] = []
        self._results = results
        self._conversation.begin(
            continuation=entry == "notice"
            or bool(self._pending_notices)
            or bool(self._quiet_notices)
            or self._watched_provider is not None
            or self._tools.confirmation is not None
            or bool(self._confirming)
        )
        if notice is not None:
            self._append_notice(notice)
        else:
            while self._pending_notices:
                pending = self._pending_notices.popleft()
                self._append_notice(pending)
                self._context_scope = replace(
                    pending.context,
                    modules=ordered_modules(
                        (*self._context_scope.modules, *pending.context.modules)
                    ),
                    message_index=pending.index,
                )
        self._append_quiet_notices()
        self._conversation.trim()
        context_index = self._context_scope.message_index
        if context_index is not None and self._messages[context_index].archived:
            self._context_scope = _ContextScope()
        decision_input = classification_input(
            text,
            recent_messages=self._conversation.active_messages,
            active_task=self._context_scope.summary,
            last_puri_action=self._last_action_summary(),
            previous_modules=self._context_scope.modules,
        )
        message_index = notice.index if notice is not None else None
        if entry != "notice":
            message_index = self._conversation.display("user", text)
            self._pending_user = text
            self._conversation.record({"role": "user", "content": text})
        task = self._scope.create_task(
            self._run_turn(
                text,
                entry=entry,
                turn_id=turn_id,
                diagnostic_id=diagnostic_id,
                generation=generation,
                decision_input=decision_input,
                notice=notice,
                message_index=message_index,
                turn_results=results,
            ),
            name=f"turn-{generation}",
        )
        self._active = task
        self._publish("thinking")
        return task

    async def _context(
        self,
        *,
        diagnostic_id: str | None = None,
        topic: str | None = None,
        operation_id: str | None = None,
        after: int | None = None,
    ) -> dict:
        arguments = {"topic": topic} if topic is not None else {}
        context = await collect_agent_context(
            self._control,
            activity=self._activity,
            localize=self._localize,
            diagnostic_id=diagnostic_id,
            operation_id=operation_id,
            after=after,
            recent_agent_actions=tuple(self._actions),
            puri_allowance_exhausted=self._allowance_exhausted,
            **arguments,
        )
        operations = list(self._operations.values())
        pending = [
            item
            for item in operations
            if item["phase"] == "submitted" or item.get("result", {}).get("status") == "pending"
        ]
        for item in pending:
            receipt = item.get("result", {}).get("receipt", item.get("receipt", {}))
            if not receipt.get("operation_id"):
                continue
            try:
                observed = await self._control.operation(receipt["operation_id"])
            except Exception:
                continue
            self._observe_operation_receipt(item, observed)
        pending = [
            item
            for item in pending
            if item["phase"] == "submitted" or item.get("result", {}).get("status") == "pending"
        ]
        completed = [item for item in operations if item not in pending][-5:]
        context["agent_operations"] = deepcopy(pending + completed)
        context["pending_confirmation"] = self._tools.confirmation
        context["support_reports"] = [
            {
                "status": report.status,
                "offer_id": report.id,
                "displayed": report.displayed,
                "clicked": report.clicked,
                **({"ticket": report.ticket} if report.status == "sent" else {}),
            }
            for report in self._reports.values()
            if report.message_index is not None
            and not self._messages[report.message_index].archived
        ][-5:]
        return context

    def _select_context(
        self, decision: AgentDecision, entry: AgentDecisionEntry, notice: _AppNotice | None
    ) -> tuple[tuple[str, ...], str, str | None]:
        if decision.route in {"social", "out_of_scope"}:
            return (), "brief", None
        if self._context_mode == "full":
            return MODULES, "full", None
        if entry == "error":
            return MODULES, "error", None
        if notice is not None:
            if notice.context.modules:
                return notice.context.modules, "notice", None
            return MODULES, "fallback", "missing_notice_context"
        if decision.context_modules is None:
            return (
                MODULES,
                "fallback",
                decision.context_reason or decision.reason or "unclear_route",
            )
        modules = decision.context_modules
        if decision.context_relation == "continue":
            modules = ordered_modules((*self._context_scope.modules, *modules))
        return modules, "model", None

    async def _run_turn(
        self,
        text: str,
        *,
        entry: AgentDecisionEntry,
        turn_id: str,
        diagnostic_id: str | None,
        generation: int,
        decision_input: list[dict],
        notice: _AppNotice | None,
        message_index: int | None,
        turn_results: list[AgentToolResult],
    ) -> AgentTurnResult:
        context_task = None
        decision_task = None
        started = self._clock()
        decision_seconds = None
        first_useful_seconds = None
        model_calls = 0
        try:
            self._require_current(generation)
            if self._allowance_exhausted:
                if entry != "notice":
                    self._append_user(text)
                return self._finish(
                    "service_unavailable",
                    self._localize("agent_chat.service_allowance_exhausted"),
                )
            if self._prompt_requests >= 15 or self._clock() - self._prompt_started >= 86400:
                raise AgentServiceError("service_turn_exhausted")
            await self._transport.prepare()
            self._require_current(generation)
            context_task = self._scope.create_task(
                self._context(diagnostic_id=diagnostic_id), name=f"context-{generation}"
            )
            if entry == "message":
                decision_started = self._clock()
                model_calls += 1
                self._prompt_requests += 1
                decision_task = self._scope.create_task(
                    self._classify(decision_input, turn_id=turn_id), name=f"decision-{generation}"
                )
                done, _ = await asyncio.wait({decision_task}, timeout=self._decision_timeout)
                self._require_current(generation)
                decision_seconds = self._clock() - decision_started
                if done:
                    decision = decision_task.result()
                    if isinstance(decision, AgentServiceError):
                        model_calls -= 1
                        self._emit_decision(
                            entry, generation, AgentDecision(source="error", reason=decision.code)
                        )
                        self._append_user(text)
                        raise decision
                else:
                    decision_task.cancel()
                    decision = AgentDecision(source="fallback", reason="classify_timeout")
            else:
                decision = AgentDecision(
                    "work",
                    "high" if entry == "error" else notice.effort,
                    source=entry,
                )
            self._require_current(generation)
            context_modules, context_source, context_reason = self._select_context(
                decision, entry, notice
            )
            self._emit_decision(
                entry,
                generation,
                decision,
                context_modules=context_modules,
                context_source=context_source,
                context_reason=context_reason,
            )
            out_of_scope = decision.route == "out_of_scope"
            brief = decision.route in {"social", "out_of_scope"}
            if brief:
                context_task.cancel()
                self._append_user(text)
            else:
                self._last_effort = decision.effort
                context = await context_task
                self._require_current(generation)
                self._tools.observe_context(context)
                previous = notice.context if notice is not None else self._context_scope
                self._context_scope = _ContextScope(
                    context_modules,
                    message_index,
                    (
                        previous.summary
                        if previous.summary is not None
                        and (notice is not None or decision.context_relation == "continue")
                        else text[:240]
                    ),
                )
                self._conversation.append(
                    {"type": "configuration_update", "reasoning": {"effort": decision.effort}},
                    archive=False,
                )
                context_view = self._conversation.project_context(
                    model_context(context, localize=self._localize, modules=context_modules)
                )
                self._conversation.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"app_context": context_view},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                    context=context_view,
                )
                if entry != "notice":
                    self._append_user(text)
            calls = 0
            while True:
                self._require_current(generation)
                self._publish("thinking")
                self._require_current(generation)
                if self._prompt_requests >= 15:
                    raise AgentServiceError("service_turn_exhausted")
                self._prompt_requests += 1
                model_calls += 1
                try:
                    stream = self._transport.respond(
                        turn_id=turn_id,
                        input=(
                            [{"role": "user", "content": text}]
                            if out_of_scope
                            else deepcopy(
                                [
                                    {"role": message.role, "content": message.text}
                                    for message in self._conversation.active_messages
                                    if message.role in {"user", "assistant"}
                                ]
                                if decision.route == "social"
                                else self._items
                            )
                        ),
                        effort=decision.effort,
                        route=decision.route,
                        allow_tools=calls < 10
                        and self._prompt_requests < 15
                        and decision.route in {"work", "unclear"},
                    )
                    presentation = AgentPresentationBuffer(generation, model_calls)
                    self._presentation = presentation
                    response = None
                    try:
                        async for event in stream:
                            self._require_current(generation)
                            if not isinstance(event, dict) or response is not None:
                                raise AgentTransportError("invalid_response")
                            if event.get("type") == "text_delta" and isinstance(
                                event.get("text"), str
                            ):
                                if presentation.append(event["text"]):
                                    if first_useful_seconds is None:
                                        first_useful_seconds = self._clock() - started
                                    self._publish("responding")
                            elif event.get("type") == "completed" and isinstance(
                                event.get("response"), dict
                            ):
                                response = event["response"]
                            else:
                                raise AgentTransportError("invalid_response")
                    finally:
                        await stream.aclose()
                    self._require_current(generation)
                    if response is None:
                        raise AgentTransportError("invalid_response")
                except AgentServiceError:
                    model_calls -= 1
                    raise
                self._require_current(generation)
                output = response.get("output")
                if not isinstance(output, list) or any(
                    not isinstance(item, dict) for item in output
                ):
                    raise AgentTransportError("invalid_response")
                function_calls = [item for item in output if item.get("type") == "function_call"]
                if (
                    response.get("status") != "completed"
                    or any(item.get("status") not in (None, "completed") for item in output)
                    or not self._valid_calls(function_calls)
                ):
                    self._presentation = None
                    return self._finish("interrupted", self._localize("agent_chat.interrupted"))
                if function_calls and self._prompt_requests >= 15:
                    raise AgentServiceError("service_turn_exhausted")
                reply = self._response_text(output)
                if not function_calls and not reply:
                    self._presentation = None
                    return self._finish("interrupted", self._localize("agent_chat.interrupted"))
                presentation.complete(reply)
                if out_of_scope:
                    if function_calls:
                        self._presentation = None
                        return self._finish("interrupted", self._localize("agent_chat.interrupted"))
                    self._conversation.append({"role": "assistant", "content": reply})
                    first_useful_seconds = self._clock() - started
                    return self._finish("completed", reply, model_answer=True)
                for item in output:
                    self._conversation.append(item)
                if function_calls:
                    self._presentation = None
                    if decision.route == "social":
                        for call in function_calls:
                            self._record_result(
                                call, None, {"status": "not_executed", "code": "social_turn"}
                            )
                        return self._finish("interrupted", self._localize("agent_chat.interrupted"))
                    if len(function_calls) > 4 or calls + len(function_calls) > 10:
                        for call in function_calls:
                            self._record_result(
                                call, None, {"status": "rejected", "code": "tool_limit"}
                            )
                        return self._finish("interrupted", self._localize("agent_chat.tool_limit"))
                    parsed_calls = [self._arguments(call) for call in function_calls]
                    calls += len(function_calls)
                    reads = []
                    for call, arguments in zip(function_calls, parsed_calls, strict=True):
                        try:
                            action = select_app_action(call["name"], arguments or {})
                            if call["name"] != "read_app" or action.handler != "context":
                                break
                            validate(arguments, action.schema({}))
                            reads.append(action.arguments(arguments))
                        except ValueError:
                            break
                    if len(reads) == len(function_calls) and len(reads) > 1:
                        self._active_tool = ("read_app", reads[0]["topic"])
                        self._publish("working")
                        contexts = await asyncio.gather(
                            *(self._context(**arguments) for arguments in reads),
                            return_exceptions=True,
                        )
                        self._require_current(generation)
                        for call, arguments, context in zip(
                            function_calls, reads, contexts, strict=True
                        ):
                            if isinstance(context, BaseException):
                                failure = {"status": "unavailable", "code": "tool_unavailable"}
                                if isinstance(context, Exception):
                                    failure["diagnostics"] = self._failure_evidence(
                                        context,
                                        code="agent_tool_failed",
                                        operation="agent_tool",
                                        stage="tool_dispatch",
                                    )
                                self._record_result(
                                    call,
                                    arguments["topic"],
                                    failure,
                                )
                            else:
                                self._record_context_result(
                                    call,
                                    arguments["topic"],
                                    context,
                                    generation,
                                    stage_controls=False,
                                )
                        self._publish("thinking")
                        continue
                    blocked = None
                    for call, arguments in zip(function_calls, parsed_calls, strict=True):
                        self._require_current(generation)
                        pure_read = call.get("name") == "read_app"
                        if blocked is not None and not pure_read:
                            self._record_result(
                                call, None, {"status": "not_executed", "code": blocked}
                            )
                            continue
                        result = await self._execute_call(call, arguments, generation)
                        self._require_current(generation)
                        status = result["status"]
                        if status == "needs_confirmation":
                            blocked = "confirmation_pending"
                        elif status in {
                            "pending",
                            "action_required",
                            "failed",
                            "rejected",
                            "unavailable",
                            "cancelled",
                            "interrupted",
                            "degraded",
                            "persistence_failed",
                            "not_executed",
                        }:
                            blocked = "read_failed" if pure_read else "previous_call_blocked"
                        elif result.get("controls_staged"):
                            blocked = "user_action_required"
                        if first_useful_seconds is None:
                            first_useful_seconds = self._clock() - started
                    self._publish("thinking")
                    continue
                if first_useful_seconds is None:
                    first_useful_seconds = self._clock() - started
                return self._finish("completed", reply, model_answer=True)
        except asyncio.CancelledError:
            if self._closed:
                raise
            if generation == self._generation:
                self._close_outstanding_calls()
            return AgentTurnResult(
                "cancelled", self._localize("agent_chat.interrupted"), tuple(turn_results)
            )
        except AgentServiceError as service_error:
            self._require_current(generation)
            if self._pending_user is not None:
                self._append_user(self._pending_user)
            if service_error.code == "service_allowance_exhausted":
                self._allowance_exhausted = True
            elif service_error.code == "service_turn_exhausted":
                self._prompt_requests = 15
            return self._finish(
                "service_unavailable",
                self._localize(f"agent_chat.{service_error.code}"),
                diagnostics=self._failure(
                    service_error, code="agent_service_failed", operation="agent_chat"
                ),
            )
        except AgentTransportError as exc:
            self._require_current(generation)
            if self._pending_user is not None:
                self._append_user(self._pending_user)
            return self._finish(
                "failed",
                self._localize("agent_chat.connection_failed"),
                diagnostics=self._failure(
                    exc, code="agent_transport_failed", operation="agent_chat"
                ),
            )
        finally:
            self._observations.append(
                AgentTurnObservation(decision_seconds, first_useful_seconds, model_calls)
            )
            if context_task is not None and not context_task.done():
                context_task.cancel()
            if decision_task is not None and not decision_task.done():
                decision_task.cancel()
            if generation == self._generation:
                for report in self._reports.values():
                    if report.generation == generation and report.status == "pending":
                        report.status = "cancelled"
                if self._presentation is not None and not self._presentation.snapshot.completed:
                    self._presentation = None
                self._active = None
                self._conversation.trim()
                self._publish("idle")
                self._schedule_notices()

    async def _execute_call(self, call: dict, arguments: dict | None, generation: int) -> dict:
        self._require_current(generation)
        name = call.get("name")
        if arguments is None:
            result = {"status": "rejected", "code": "invalid_tool_call"}
            self._record_result(call, None, result)
            return result
        try:
            action = select_app_action(name, arguments)
            if not action.settings:
                validate(arguments, action.schema({}))
        except ValueError:
            result = {
                "status": "rejected",
                "code": "invalid_tool_call",
                "explanation": "Choose a supported action and supply only its applicable fields; omit nulls.",
            }
            self._record_result(call, None, result)
            return result
        topic = arguments[action.key]
        bound = action.arguments(arguments)
        stage_controls = name == "open_app"
        self._active_tool = (name, topic)
        self._publish("working")
        try:
            if action.handler == "context":
                context = await self._context(**bound)
                self._require_current(generation)
                return self._record_context_result(
                    call, bound["topic"], context, generation, stage_controls=stage_controls
                )
            if action.handler == "operation":
                identity = bound["operation_id"]
                operation = next(
                    (
                        item
                        for item in self._operations.values()
                        if item.get("result", {})
                        .get("receipt", item.get("receipt", {}))
                        .get("operation_id")
                        == identity
                    ),
                    None,
                )
                if operation is None:
                    result = {"status": "not_retained", "operation_id": identity}
                else:
                    result = deepcopy(
                        operation.get(
                            "result", {"status": "pending", "receipt": operation.get("receipt", {})}
                        )
                    )
                    if result.get("status") == "pending":
                        receipt = await self._control.operation(identity)
                        self._require_current(generation)
                        if self._observe_operation_receipt(operation, receipt):
                            result = deepcopy(operation["result"])
            elif action.handler == "report":
                result = self._propose_report(bound["summary"], generation)
            else:
                result = await self._tools.execute(
                    name,
                    arguments,
                    require_current=lambda: self._require_current(generation),
                    request_id=call["call_id"],
                )
                self._require_current(generation)
                if (
                    action.handler == "provider_setup"
                    and result.get("status") == "available"
                    and stage_controls
                ):
                    self._stage_provider_setup(bound["provider"], generation)
                    result = {**result, "controls_staged": True, "displayed": False}
                if action.handler == "navigation" and result.get("opened") is True:
                    result = {**result, "controls_staged": True}
                if isinstance(result.get("context"), dict):
                    self._tools.observe_context(result["context"])
        except Exception as exc:
            result = {
                "status": "unavailable",
                "code": "tool_unavailable",
                "diagnostics": self._failure_evidence(
                    exc, code="agent_tool_failed", operation="agent_tool", stage="tool_dispatch"
                ),
            }
        self._require_current(generation)
        projected = model_receipt(result)
        self._record_result(call, topic, projected)
        return projected

    def _observe_operation_receipt(self, operation: dict, receipt: dict) -> bool:
        retained = operation.get("result", {}).get("receipt", operation.get("receipt", {}))
        if (
            self._operations.get(operation["action_id"]) is not operation
            or not isinstance(receipt, dict)
            or not retained.get("operation_id")
            or receipt.get("operation_id") != retained["operation_id"]
            or receipt.get("request_id") != operation["action_id"]
            or not isinstance(receipt.get("status"), str)
        ):
            return False
        operation["result"] = {
            "status": receipt["status"] if receipt.get("terminal") is True else "pending",
            "receipt": project_operation_receipt(receipt, operation["tool"]),
        }
        if receipt.get("terminal") is True:
            operation["phase"] = "completed"
            self._tools.observe_operation_receipt(receipt)
        return True

    def _require_notice_epoch(self, epoch: int) -> None:
        if self._closed or epoch != self._notice_epoch:
            raise asyncio.CancelledError

    def _record_action(self, action: dict) -> None:
        identity = action["action_id"]
        self._operations[identity] = {**self._operations.get(identity, {}), **deepcopy(action)}
        self._conversation.record({"type": "app_operation", **action})
        result = action.get("result", {})
        if action.get("changed") and result.get("status") in {"applied", "degraded"}:
            self._actions.append(deepcopy(action))

    def _append_user(self, text: str) -> None:
        self._pending_user = None
        self._conversation.append({"role": "user", "content": text}, archive=False)

    async def _classify(
        self, input: list[dict], *, turn_id: str
    ) -> AgentDecision | AgentServiceError:
        try:
            return extract_decision(await self._transport.classify(input=input, turn_id=turn_id))
        except AgentServiceError as error:
            return error
        except AgentTransportError:
            return AgentDecision(source="fallback", reason="classify_transport_error")

    def _emit_decision(
        self,
        entry: AgentDecisionEntry,
        generation: int,
        decision: AgentDecision,
        *,
        context_modules: tuple[str, ...] = (),
        context_source: str = "unavailable",
        context_reason: str | None = None,
    ) -> None:
        self._emit_event(
            {
                "kind": "decision",
                "generation": generation,
                "entry": entry,
                "source": decision.source,
                "route": decision.route,
                "effort": decision.effort,
                "reason": decision.reason,
                "context_mode": self._context_mode,
                "context_modules": list(context_modules),
                "context_source": context_source,
                "context_reason": context_reason,
                "context_relation": decision.context_relation,
            }
        )

    def _last_action_summary(self) -> str | None:
        if not self._actions:
            return None
        action = self._actions[-1]
        result = action.get("result")
        status = result.get("status") if isinstance(result, dict) else action.get("status")
        return " / ".join(
            value
            for value in (action.get("tool"), action.get("topic"), action.get("phase"), status)
            if isinstance(value, str)
        )[:240]

    def _require_current(self, generation: int) -> None:
        if self._closed or generation != self._generation:
            raise asyncio.CancelledError

    def _finish(
        self,
        status: str,
        text: str,
        *,
        model_answer: bool = False,
        diagnostics: ErrorDiagnostics | None = None,
    ) -> AgentTurnResult:
        index = self._conversation.display("assistant" if model_answer else "notice", text)
        if model_answer:
            response_index = (
                self._presentation.snapshot.response_index
                if self._presentation is not None
                else None
            )
            for report in self._reports.values():
                if report.generation == self._generation and report.status == "pending":
                    report.message_index = index
                    report.response_index = response_index
                    report.status = "offered"
            for identity, action in tuple(self._setup_actions.items()):
                if (
                    action.generation == self._generation
                    and action.message_index is None
                    and action.status != "cancelled"
                ):
                    self._setup_actions[identity] = replace(
                        action,
                        response_index=response_index,
                        message_index=index,
                        available=self._setup_action_handler is not None,
                        status="available",
                    )
            for identity, request in tuple(self._secure_requests.items()):
                if request.generation == self._generation and request.status == "pending":
                    self._secure_requests[identity] = replace(
                        request,
                        response_index=response_index,
                        message_index=index,
                        available=self._secure_entry is not None,
                        status="ready",
                    )
        if not model_answer:
            self._conversation.append({"role": "assistant", "content": text})
        return AgentTurnResult(status, text, tuple(self._results), diagnostics)

    def _valid_calls(self, calls: list[dict]) -> bool:
        ids = [call.get("call_id") for call in calls]
        prior = {item.get("call_id") for item in self._items if item.get("type") == "function_call"}
        return (
            all(isinstance(identity, str) and identity for identity in ids)
            and len(ids) == len(set(ids))
            and not prior.intersection(ids)
            and all(call.get("status") == "completed" for call in calls)
        )

    @staticmethod
    def _arguments(call: dict) -> dict | None:
        if call.get("name") not in APP_TOOLS or not isinstance(call.get("arguments"), str):
            return None
        try:
            arguments = json.loads(call["arguments"])
        except ValueError:
            return None
        return arguments if isinstance(arguments, dict) else None

    def _record_context_result(
        self, call: dict, topic: str, context: dict, generation: int, *, stage_controls: bool = True
    ) -> dict:
        self._require_current(generation)
        self._tools.observe_context(context)
        staged = (
            stage_controls
            and topic == "accounts"
            and self._stage_account_actions(context, generation)
        )
        view = self._conversation.project_context(
            model_context(context, topic, self._localize),
            force_modules=("accounts",) if staged else (),
        )
        unchanged = (
            not view.get("missing")
            and not view.get("catalogs")
            and view.get("unchanged") == view["modules"]
        )
        result = {
            "status": "unchanged" if unchanged else "observed",
            "context": view,
            **({"controls_staged": True, "displayed": False} if staged else {}),
        }
        self._record_result(call, topic, result)
        return result

    def _record_result(self, call: dict, topic: str | None, result: dict) -> None:
        context = None
        if isinstance(result.get("context"), dict) and result.get("status") in {
            "observed",
            "unchanged",
        }:
            context = result["context"]
            modules = context["modules"]
            self._context_scope = replace(
                self._context_scope,
                modules=ordered_modules((*self._context_scope.modules, *modules)),
            )
        card = result.get("confirmation")
        if result.get("status") == "needs_confirmation" and isinstance(card, dict):
            self._card_context = (
                card["id"],
                replace(
                    self._context_scope,
                    modules=ordered_modules(
                        (*self._context_scope.modules, *confirmation_modules(card))
                    ),
                ),
            )
        self._conversation.append(
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            },
            context=context,
        )
        name = call.get("name", "unknown")
        self._results.append(
            AgentToolResult(call["call_id"], topic, result["status"], result, name)
        )

    def _close_outstanding_calls(self) -> None:
        paired = {
            item["call_id"] for item in self._items if item.get("type") == "function_call_output"
        }
        for item in tuple(self._items):
            if item.get("type") == "function_call" and item["call_id"] not in paired:
                self._record_result(item, None, {"status": "cancelled", "code": "turn_cancelled"})

    @staticmethod
    def _response_text(output: list[dict]) -> str:
        parts = []
        for item in output:
            if item.get("type") == "message" and item.get("role") == "assistant":
                content_items = item.get("content", [])
                if not isinstance(content_items, list):
                    continue
                for content in content_items:
                    if isinstance(content, dict) and content.get("type") in {
                        "output_text",
                        "refusal",
                    }:
                        text = content.get("text", content.get("refusal"))
                        if isinstance(text, str):
                            parts.append(text)
        return "\n".join(parts)
