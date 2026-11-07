from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from puripuly_heart.app.ports.agent_chat import (
    AgentChatSnapshot,
    AgentReportOffer,
    AgentSecureEntryRequest,
    AgentSetupAction,
    AgentToolResult,
)
from puripuly_heart.app.services.agent_chat.presentation import completed_sentences
from puripuly_heart.ui.i18n import language_name, provider_label, t

APP_NOTICE_PREFIX = "[app notice]"
MANUAL_RESET = "새 대화로 시작했어요"
RESET_SEPARATORS = frozenset({MANUAL_RESET, "30분이 지나 새 대화로 시작했어요"})
MAX_CHANGE_BLOCKS = 2
LONG_TEXT_LIMIT = 24

_CONTEXT_SUBJECT_KEYS = {
    "overview": "agent_chat.subject.overview",
    "settings": "agent_chat.subject.settings",
    "translation": "agent_chat.subject.translation",
    "output": "agent_chat.subject.output",
    "audio": "agent_chat.subject.audio",
    "accounts": "agent_chat.subject.accounts",
    "diagnostics": "agent_chat.subject.diagnostics",
}
_TOOL_STATUS_KEYS = {
    "diagnose_microphone": "agent_chat.status.microphone",
    "probe_self_recognition": "agent_chat.status.recognition_probe",
    "probe_peer_recognition": "agent_chat.status.recognition_probe",
    "probe_translation": "agent_chat.status.translation_probe",
    "change_settings": "agent_chat.status.change_settings",
    "open_settings": "agent_chat.status.open_settings",
    "get_provider_setup": "agent_chat.status.provider_setup",
    "retry_osc_discovery": "agent_chat.status.osc",
    "set_capture": "agent_chat.status.capture",
    "set_translation": "agent_chat.status.translation_toggle",
}
_QUIET_TOOLS = frozenset({"withdraw_confirmation", "propose_report"})
_NOT_DONE_STATUSES = frozenset(
    {
        "rejected",
        "failed",
        "cancelled",
        "unavailable",
        "not_executed",
        "needs_confirmation",
        "declined",
        "withdrawn",
        "pending",
        "action_required",
    }
)

_FIELD_LABEL_KEYS = {
    "locale": "agent_chat.field.locale",
    "audio": "agent_chat.field.microphone",
    "peer_expected_languages": "agent_chat.field.peer_expected_languages",
    "self_vad.speech_threshold": "agent_chat.field.self_vad_threshold",
    "peer_vad.speech_threshold": "agent_chat.field.peer_vad_threshold",
    "peer_vad.hangover_ms": "agent_chat.field.peer_vad_hangover",
    "peer_vad.pre_roll_ms": "agent_chat.field.peer_vad_pre_roll",
    "overlay.target": "agent_chat.field.overlay_target",
    "overlay.translation.enabled": "agent_chat.field.overlay_translation",
    "overlay.peer_original.enabled": "agent_chat.field.overlay_peer_original",
    "overlay.background_alpha": "agent_chat.field.overlay_background",
    "overlay.swap_caption_languages": "agent_chat.field.overlay_swap",
    "stt.provider": "agent_chat.field.self_recognition",
    "stt.cloud_free_tier_providers": "agent_chat.field.free_recognition",
    "peer_stt.provider": "agent_chat.field.peer_recognition",
    "translation": "agent_chat.field.translation",
    "languages.source": "agent_chat.field.source_language",
    "languages.target": "agent_chat.field.target_language",
    "languages.secondary_target": "agent_chat.field.secondary_target_language",
    "languages.peer_source": "agent_chat.field.peer_source_language",
    "languages.peer_target": "agent_chat.field.peer_target_language",
    "languages.peer_source_mode": "agent_chat.field.peer_source_mode",
}
_PERCENT_FIELDS = frozenset(
    {"self_vad.speech_threshold", "peer_vad.speech_threshold", "overlay.background_alpha"}
)
_MS_FIELDS = frozenset({"peer_vad.hangover_ms", "peer_vad.pre_roll_ms"})
_PROVIDER_FIELDS = frozenset({"stt.provider", "peer_stt.provider"})
_LANGUAGE_FIELDS = frozenset(
    {
        "languages.source",
        "languages.target",
        "languages.secondary_target",
        "languages.peer_source",
        "languages.peer_target",
    }
)


@dataclass(frozen=True, slots=True)
class StatusLine:
    text: str
    busy: bool


@dataclass(frozen=True, slots=True)
class ChangeBlock:
    label: str
    before: str
    after: str
    long_text: bool = False


LeftKind = Literal["home", "history", "change", "key_create", "key_entry", "report", "account"]


@dataclass(frozen=True, slots=True)
class LeftView:
    kind: LeftKind
    key: str = "home"
    setup_action: AgentSetupAction | None = None
    secure_request: AgentSecureEntryRequest | None = None
    report: AgentReportOffer | None = None
    confirmation: dict | None = None


ChatKind = Literal["greeting", "user", "assistant", "system", "reset", "status"]


@dataclass(frozen=True, slots=True)
class ChatItem:
    kind: ChatKind
    key: str
    lines: tuple[str, ...] = ()
    archived: bool = False
    busy: bool = False


def tool_label(result: AgentToolResult | tuple[str, str | None], *, done: bool) -> str | None:
    tool, topic = (result.tool, result.topic) if isinstance(result, AgentToolResult) else result
    if tool in _QUIET_TOOLS:
        return None
    suffix = "done" if done else "active"
    if tool == "get_app_context":
        subject = t(
            _CONTEXT_SUBJECT_KEYS.get(topic or "overview", _CONTEXT_SUBJECT_KEYS["overview"])
        )
        return t(f"agent_chat.status.check.{suffix}", subject=subject)
    key = _TOOL_STATUS_KEYS.get(tool)
    return t(f"{key}.{suffix}") if key is not None else None


def active_status(snapshot: AgentChatSnapshot) -> StatusLine | None:
    if not snapshot.busy:
        return None
    active = getattr(snapshot, "active_tool", None)
    if active is not None:
        label = tool_label(active, done=False)
        if label:
            return StatusLine(label, True)
    if snapshot.phase == "responding":
        done = done_status(snapshot.tool_results)
        return StatusLine(done, False) if done else None
    return StatusLine(t("agent_chat.status.thinking"), True)


def done_status(results: Sequence[AgentToolResult]) -> str | None:
    subjects: list[str] = []
    others: list[str] = []
    for result in results:
        if result.status in _NOT_DONE_STATUSES or result.tool in _QUIET_TOOLS:
            continue
        if result.tool == "get_app_context":
            subject = t(
                _CONTEXT_SUBJECT_KEYS.get(
                    result.topic or "overview", _CONTEXT_SUBJECT_KEYS["overview"]
                )
            )
            if subject not in subjects:
                subjects.append(subject)
            continue
        if result.tool == "change_settings":
            changes = result.result.get("changes")
            label = applied_label(changes) if isinstance(changes, dict) else None
        else:
            label = tool_label(result, done=True)
        if label and label not in others:
            others.append(label)
    parts = []
    if subjects:
        parts.append(t("agent_chat.status.check.done", subject=" · ".join(subjects)))
    parts.extend(others)
    return " · ".join(parts) if parts else None


def applied_label(changes: dict) -> str | None:
    labels = [block.label for block in change_blocks({"changes": changes, "previous": {}})]
    if not labels:
        return None
    if len(labels) == 1:
        return t("agent_chat.status.changed_one", label=labels[0])
    return t("agent_chat.status.changed_many", label=labels[0], count=len(labels) - 1)


def _value_text(field: str, value: object) -> tuple[str, bool]:
    if value is None or value == "":
        return t("agent_chat.value.none"), False
    if isinstance(value, bool):
        return t("agent_chat.value.on" if value else "agent_chat.value.off"), False
    if field in _PERCENT_FIELDS and isinstance(value, int | float):
        return f"{round(float(value) * 100)}%", False
    if field in _MS_FIELDS and isinstance(value, int | float):
        return f"{int(value)}ms", False
    if field in _PROVIDER_FIELDS and isinstance(value, str):
        return provider_label(value), False
    if field in _LANGUAGE_FIELDS and isinstance(value, str):
        return language_name(value), False
    if field == "locale" and isinstance(value, str):
        return t(f"locale.{value}", default=value), False
    if isinstance(value, list | tuple):
        return t("agent_chat.value.count", count=len(value)), False
    if isinstance(value, dict):
        return t("agent_chat.value.count", count=len(value)), False
    text = str(value)
    return text, len(text) > LONG_TEXT_LIMIT


def _translation_value(change: dict) -> str:
    model = change.get("translation.model")
    connection = change.get("translation.connection")
    parts = []
    if isinstance(model, str):
        parts.append(t(f"provider.{model}", default=model))
    if isinstance(connection, str):
        parts.append(t(f"settings.translation_connection.{connection}", default=connection))
    return " · ".join(parts) or t("agent_chat.value.none")


def _audio_value(change: dict) -> str:
    device = change.get("audio.input_device")
    host = change.get("audio.input_host_api")
    if isinstance(device, str) and device:
        return device
    if isinstance(host, str) and host:
        return host
    return t("agent_chat.value.default_device")


def change_blocks(card: dict) -> list[ChangeBlock]:
    changes = card.get("changes") or {}
    previous = card.get("previous") or {}
    blocks: list[ChangeBlock] = []
    if not isinstance(changes, dict):
        return blocks
    if "translation.model" in changes or "translation.connection" in changes:
        blocks.append(
            ChangeBlock(
                t(_FIELD_LABEL_KEYS["translation"]),
                _translation_value(previous) if previous else "",
                _translation_value(changes),
            )
        )
    if "audio.input_host_api" in changes or "audio.input_device" in changes:
        blocks.append(
            ChangeBlock(
                t(_FIELD_LABEL_KEYS["audio"]),
                _audio_value(previous) if previous else "",
                _audio_value(changes),
            )
        )
    for field, value in changes.items():
        if field.startswith(("translation.", "audio.")):
            continue
        if field == "languages" and isinstance(value, dict):
            before_languages = (
                previous.get("languages") if isinstance(previous.get("languages"), dict) else {}
            )
            for name, language in value.items():
                path = f"languages.{name}"
                label_key = _FIELD_LABEL_KEYS.get(path)
                if label_key is None:
                    continue
                after, _ = _value_text(path, language)
                before = (
                    _value_text(path, before_languages[name])[0] if name in before_languages else ""
                )
                blocks.append(ChangeBlock(t(label_key), before, after))
            continue
        label_key = _FIELD_LABEL_KEYS.get(field)
        if label_key is None:
            continue
        after, long_text = _value_text(field, value)
        before = _value_text(field, previous[field])[0] if field in previous else ""
        if long_text and field in previous and previous[field] in (None, ""):
            before = t("agent_chat.value.default")
        blocks.append(ChangeBlock(t(label_key), before, after, long_text))
    return blocks


def _current_setup(snapshot: AgentChatSnapshot) -> AgentSetupAction | None:
    for action in reversed(snapshot.setup_actions):
        if action.available and action.message_index is not None and action.kind == "key_console":
            return action
    return None


def _current_account(snapshot: AgentChatSnapshot) -> AgentSetupAction | None:
    for action in reversed(snapshot.setup_actions):
        if (
            action.available
            and action.message_index is not None
            and action.kind in {"account_setup", "public_usage"}
        ):
            return action
    return None


def _current_secure(snapshot: AgentChatSnapshot) -> AgentSecureEntryRequest | None:
    for request in reversed(snapshot.secure_entry_requests):
        if request.available and request.message_index is not None:
            return request
    return None


def _current_report(snapshot: AgentChatSnapshot) -> AgentReportOffer | None:
    for offer in reversed(snapshot.report_offers):
        if offer.message_index is not None and offer.status in {
            "offered",
            "sending",
            "sent",
            "failed",
        }:
            return offer
    return None


def _response_settled(snapshot: AgentChatSnapshot) -> bool:
    presentation = snapshot.presentation
    return presentation is None or presentation.completed


def select_left(
    snapshot: AgentChatSnapshot,
    *,
    history_open: bool = False,
    key_entry_step: set[str] | frozenset[str] = frozenset(),
    dismissed: set[str] | frozenset[str] = frozenset(),
) -> LeftView:
    settled = _response_settled(snapshot)
    secure = _current_secure(snapshot)
    if settled and secure is not None and f"secure:{secure.id}" not in dismissed:
        setup = next(
            (
                action
                for action in reversed(snapshot.setup_actions)
                if action.secure_request_id == secure.id and action.kind == "key_console"
            ),
            None,
        )
        if setup is not None and setup.available and secure.id not in key_entry_step:
            return LeftView(
                "key_create", f"key_create:{setup.id}", setup_action=setup, secure_request=secure
            )
        return LeftView(
            "key_entry", f"key_entry:{secure.id}", setup_action=setup, secure_request=secure
        )
    card = snapshot.confirmation
    if card is not None:
        return LeftView("change", f"change:{card['id']}", confirmation=card)
    report = _current_report(snapshot)
    if settled and report is not None and f"report:{report.id}" not in dismissed:
        return LeftView("report", f"report:{report.id}:{report.status}", report=report)
    account = _current_account(snapshot)
    if settled and account is not None and f"account:{account.id}" not in dismissed:
        return LeftView("account", f"account:{account.id}", setup_action=account)
    if settled and _current_setup(snapshot) is not None:
        setup = _current_setup(snapshot)
        if setup is not None and f"setup:{setup.id}" not in dismissed:
            return LeftView("key_create", f"key_create:{setup.id}", setup_action=setup)
    if history_open:
        return LeftView("history", "history")
    return LeftView("home", "home")


def chat_items(
    snapshot: AgentChatSnapshot,
    *,
    turn_lines: dict[int, str],
    sentence_splitter: Callable[[str], tuple[str, ...]] | None = None,
) -> list[ChatItem]:
    split = sentence_splitter or (lambda text: completed_sentences(text, final=True) or (text,))
    items: list[ChatItem] = [ChatItem("greeting", "greeting:0", _greeting())]
    for index, message in enumerate(snapshot.messages):
        if message.role == "notice":
            text = message.text
            if text in RESET_SEPARATORS:
                items = [
                    replace(item, archived=True) if item.kind == "greeting" else item
                    for item in items
                ]
                items.append(ChatItem("reset", f"reset:{index}", (t("agent_chat.reset_line"),)))
                if text != MANUAL_RESET:
                    items.append(ChatItem("greeting", f"greeting:{index}", _greeting()))
                continue
            if text.startswith(APP_NOTICE_PREFIX):
                continue
            if index in turn_lines:
                items.append(
                    ChatItem("status", f"status:{index}", (turn_lines[index],), message.archived)
                )
            items.append(ChatItem("system", f"system:{index}", (text,), message.archived))
            continue
        if message.role == "user":
            items.append(ChatItem("user", f"user:{index}", (message.text,), message.archived))
            continue
        if index in turn_lines:
            items.append(
                ChatItem("status", f"status:{index}", (turn_lines[index],), message.archived)
            )
        items.append(
            ChatItem("assistant", f"assistant:{index}", split(message.text), message.archived)
        )
    if not snapshot.enabled:
        items.append(ChatItem("system", "system:unavailable", (t("agent_chat.unavailable"),)))
    status = active_status(snapshot)
    if status is not None:
        items.append(ChatItem("status", "status:live", (status.text,), busy=status.busy))
    presentation = snapshot.presentation
    if presentation is not None and not presentation.completed and presentation.sentences:
        items.append(
            ChatItem(
                "assistant",
                f"assistant:live:{presentation.generation}:{presentation.response_index}",
                presentation.sentences,
            )
        )
    return items


def _greeting() -> tuple[str, ...]:
    return (
        t("agent_chat.greeting.first"),
        t("agent_chat.greeting.second"),
        t("agent_chat.greeting.third"),
    )


def last_reply_index(snapshot: AgentChatSnapshot) -> int | None:
    for index in range(len(snapshot.messages) - 1, -1, -1):
        message = snapshot.messages[index]
        if message.role == "user":
            return None
        if message.role == "assistant" or not message.text.startswith(APP_NOTICE_PREFIX):
            return index
    return None


@dataclass(frozen=True, slots=True)
class HistoryRow:
    mine: bool
    source: str | None
    translations: tuple[str, ...]
    failed: bool
    time_label: str
    source_language: str | None = None
    translation_languages: tuple[str | None, ...] = ()
    key: str = ""


def history_rows(records: Sequence[object]) -> list[HistoryRow]:
    groups: dict[object, dict] = {}
    for position, record in enumerate(records):
        channel = getattr(record, "channel", "self")
        utterance = getattr(record, "parent_utterance_id", None) or getattr(
            record, "utterance_id", "legacy"
        )
        key = ("row", position) if utterance == "legacy" else (channel, utterance)
        group = groups.setdefault(
            key,
            {
                "mine": channel == "self",
                "sources": [],
                "source_language": None,
                "translations": [],
                "translation_languages": [],
                "failed": False,
                "time": getattr(record, "timestamp_label", "") or "",
                "key": ":".join(str(part) for part in key),
            },
        )
        source = getattr(record, "source_text", None)
        translation = getattr(record, "translated_text", None)
        if source and source not in group["sources"]:
            group["sources"].append(source)
            group["source_language"] = group["source_language"] or getattr(
                record, "source_language", None
            )
        if translation and translation not in group["translations"]:
            group["translations"].append(translation)
            group["translation_languages"].append(getattr(record, "target_language", None))
        if not translation and (
            getattr(record, "disposition", None) == "failed"
            or getattr(record, "turn_status", None) == "failed"
        ):
            group["failed"] = True
    return [
        HistoryRow(
            group["mine"],
            " ".join(group["sources"]) or None,
            tuple(group["translations"]),
            group["failed"] and not group["translations"],
            group["time"],
            group["source_language"],
            tuple(group["translation_languages"]),
            group["key"],
        )
        for group in groups.values()
    ]
