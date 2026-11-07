from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

AGENT_CHAT_PROTOCOL = 5

AgentEffort = Literal["low", "high", "xhigh"]
AgentRoute = Literal["work", "social", "out_of_scope", "unclear"]
AgentDecisionEntry = Literal["message", "error", "notice"]
AgentContextMode = Literal["selected", "full"]
AgentReportStatus = Literal[
    "pending", "offered", "rejected", "cancelled", "sending", "sent", "failed"
]


class AgentTextDelta(TypedDict):
    type: Literal["text_delta"]
    text: str


class AgentCompleted(TypedDict):
    type: Literal["completed"]
    response: dict


AgentResponseEvent = AgentTextDelta | AgentCompleted


class AgentResponseStream(Protocol):
    def __aiter__(self) -> AsyncIterator[AgentResponseEvent]: ...

    async def __anext__(self) -> AgentResponseEvent: ...

    async def aclose(self) -> None: ...


class AgentTransportError(RuntimeError):
    def __init__(self, code: str = "relay_unavailable") -> None:
        self.code = code
        super().__init__(code)


class AgentServiceError(AgentTransportError):
    def __init__(
        self,
        code: Literal[
            "service_authentication_failed",
            "service_authentication_unavailable",
            "service_rate_limited",
            "service_turn_exhausted",
            "service_busy",
            "request_too_large",
            "service_allowance_exhausted",
            "service_shutdown",
            "unsupported_protocol",
        ],
    ) -> None:
        super().__init__(code)


class AgentTokenPort(Protocol):
    async def get_token(self) -> str: ...

    def invalidate(self, token: str) -> None: ...

    async def close(self) -> None: ...


class AgentTransport(Protocol):
    async def prepare(self) -> None: ...

    async def classify(self, *, input: list[dict], turn_id: str) -> dict: ...

    def respond(
        self,
        *,
        input: list[dict],
        turn_id: str,
        effort: AgentEffort,
        route: AgentRoute,
        allow_tools: bool,
    ) -> AgentResponseStream: ...

    async def upload_support(
        self,
        *,
        bundle: bytes,
        ticket: str,
        app_version: str,
        diagnostic_codes: tuple[str, ...],
    ) -> dict: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentMessage:
    role: Literal["user", "assistant", "notice"]
    text: str
    archived: bool = False


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    call_id: str
    topic: str | None
    status: str
    result: dict
    tool: str = "get_app_context"


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    status: str
    text: str
    tool_results: tuple[AgentToolResult, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentReportOffer:
    id: str
    message_index: int | None
    status: AgentReportStatus
    ticket: str | None = None
    generation: int = 0
    response_index: int | None = None
    displayed: bool = False
    clicked: bool = False
    disclosure: tuple[str, ...] = (
        "conversation_archive",
        "current_app_log",
        "recognized_and_translated_sentences",
        "user_click_required",
    )


@dataclass(frozen=True, slots=True)
class AgentPresentation:
    generation: int
    response_index: int
    sentences: tuple[str, ...]
    completed: bool = False


@dataclass(frozen=True, slots=True)
class AgentSetupAction:
    id: str
    action_id: str
    provider: str
    kind: str
    generation: int
    response_index: int | None = None
    message_index: int | None = None
    available: bool = False
    displayed: bool = False
    clicked: bool = False
    status: str = "pending"
    secure_request_id: str | None = None


@dataclass(frozen=True, slots=True)
class AgentSecureEntryRequest:
    id: str
    provider: str
    task_id: str
    generation: int
    response_index: int | None = None
    message_index: int | None = None
    masked: bool = True
    available: bool = False
    displayed: bool = False
    status: str = "pending"
    code: str | None = None
    settings_alternative: bool = True
    submit_action: str = "save_and_verify"
    cancel_action: str = "cancel"


class AgentSecureEntryPort(Protocol):
    async def submit(
        self,
        request: AgentSecureEntryRequest,
        value: str,
        *,
        is_current: Callable[[], bool],
        complete: Callable[[str, bool, str | None], None],
    ) -> dict: ...

    def cancel(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentChatSnapshot:
    enabled: bool
    busy: bool
    phase: str
    messages: tuple[AgentMessage, ...]
    tool_results: tuple[AgentToolResult, ...]
    confirmation: dict | None = None
    recent_changes: tuple[dict, ...] = ()
    report_offers: tuple[AgentReportOffer, ...] = ()
    presentation: AgentPresentation | None = None
    setup_actions: tuple[AgentSetupAction, ...] = ()
    secure_entry_requests: tuple[AgentSecureEntryRequest, ...] = ()
    confirmation_previous: dict | None = None
    active_tool: tuple[str, str | None] | None = None


@dataclass(frozen=True, slots=True)
class AgentTurnObservation:
    decision_seconds: float | None
    first_useful_seconds: float | None
    model_calls: int


class AgentChatPort(Protocol):
    @property
    def snapshot(self) -> AgentChatSnapshot: ...

    def observe(self, callback: Callable[[AgentChatSnapshot], None]) -> Callable[[], None]: ...

    def bind_navigation(self, callback: Callable[[str, str | None], dict] | None) -> None: ...

    def bind_verification_providers(self, providers: frozenset[str]) -> None: ...

    def key_verification_completed(
        self, provider: str, success: bool, code: str | None = None
    ) -> None: ...

    def bind_setup_actions(self, callback: Callable[[str], dict] | None) -> None: ...

    def bind_secure_entry(self, controller: AgentSecureEntryPort | None) -> None: ...

    def mark_setup_action_displayed(self, action_id: str) -> dict: ...

    def activate_setup_action(self, action_id: str) -> dict: ...

    def mark_secure_entry_displayed(self, request_id: str) -> dict: ...

    async def submit_secure_entry(self, request_id: str, value: str) -> dict: ...

    def cancel_secure_entry(self, request_id: str) -> dict: ...

    async def open_secure_entry_settings(self, request_id: str) -> dict: ...

    def observe_events(self, callback: Callable[[dict], bool]) -> Callable[[], None]: ...

    async def open_secure_settings(self) -> dict: ...

    async def confirm(self, card_id: str, approved: bool) -> dict: ...

    def cancel_confirmation(self) -> None: ...

    def cancel(self) -> None: ...

    def new_conversation(self) -> None: ...

    @property
    def report_archive(self) -> tuple[dict, ...]: ...

    def archive_support_report(self, text: str) -> None: ...

    async def send_report(self, offer_id: str) -> dict: ...

    def mark_report_displayed(self, offer_id: str) -> dict: ...

    def reject_report(self, offer_id: str) -> dict: ...

    async def send(self, text: str, *, diagnostic_id: str | None = None) -> AgentTurnResult: ...

    async def start(self) -> None: ...

    def freeze_ingress(self) -> None: ...

    async def close(self) -> None: ...
