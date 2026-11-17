from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from html import escape
from pathlib import Path

_IDENTITY_FIELDS = (
    "format",
    "controlled_smoke",
    "agent_protocol",
    "relay_build",
    "source_tree",
    "source_revision",
    "source_sha256",
    "campaign_plan_sha256",
    "plan_sha256",
    "plan_hash",
    "context_sha256",
    "judge_revision",
    "judge_model",
    "judge_effort",
    "user_model",
    "user_effort",
    "limits",
    "required",
    "advisory",
    "split_decision",
    "experiment_id",
    "arm_id",
    "M_provider",
    "M_model",
    "D_provider",
    "D_model",
    "adapter_sha256",
    "experimental_adapter_sha256",
    "arm_plan_sha256",
    "provider_identity",
)
_PRIVATE_FIELDS = {
    "authorization",
    "proxy_authorization",
    "auth",
    "authentication",
    "credentials",
    "credential",
    "secrets",
    "password",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "private_key",
    "secret",
    "token",
    "client_secret",
    "cookie",
    "set_cookie",
    "encrypted_content",
    "encrypted_reasoning",
    "reasoning",
    "chain_of_thought",
    "reasoning_content",
    "reasoning_text",
    "encrypted_reasoning_content",
    "thinking",
    "cot",
    "request",
    "request_body",
    "native_request",
    "native_model_request",
    "body",
    "instructions",
    "input",
    "state_before",
    "visible_before",
    "visible_after",
    "relay_contexts",
    "subscription_wire",
}
_STATUS_LABELS = {
    "passed": "통과",
    "pass": "통과",
    "failed": "실패",
    "fail": "실패",
    "blocked": "차단",
    "invalid": "무효",
    "invalid_judgment": "심사 무효",
    "harness_error": "하네스 오류",
    "cancelled": "취소",
    "incomplete": "미완료",
    "pending": "대기",
    "valid": "유효",
    "completed": "완료",
    "na": "해당 없음",
    "applied": "적용",
    "needs_confirmation": "승인 필요",
    "accepted": "접수",
    "observed": "관측",
    "missing_result": "결과 미기록",
    "unknown": "상태 미기록",
}
_STYLE = """
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #172334; background: #f3f5f8; }
body { max-width: 1120px; margin: auto; padding: 24px; line-height: 1.65; }
h1, h2, h3, h4 { line-height: 1.35; overflow-wrap: anywhere; }
a { color: #174ca1; } nav ul { padding-left: 24px; } nav li { margin: 5px 0; }
section, article { background: white; border: 1px solid #ced7e2; border-radius: 10px; padding: 18px; margin: 20px 0; }
article article { border-left: 4px solid #9babc1; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #d1d9e3; padding: 9px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }
th { background: #eef2f7; } .table-scroll { overflow-x: auto; }
pre, .literal { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; font: inherit; margin: 8px 0; }
pre { font-family: ui-monospace, monospace; font-size: .88rem; background: #f4f6f9; padding: 12px; border-radius: 5px; }
details { margin: 12px 0; } summary { cursor: pointer; font-weight: 600; }
.message { margin: 12px 0; padding: 12px 16px; border-radius: 8px; border-left: 5px solid; }
.user { background: #e9f1ff; border-color: #3769ae; margin-right: 8%; }
.assistant { background: #e8f5ee; border-color: #287951; margin-left: 8%; }
.control, .notice { background: #fff4dc; border-color: #b67715; }
.other { background: #f0edf5; border-color: #79638e; }
.status { display: inline-block; border: 1px solid #7b8797; border-radius: 5px; padding: 1px 7px; background: #f1f3f6; }
.good { color: #125133; background: #e5f5e9; } .bad { color: #8b1d1d; background: #ffeded; }
.warning { padding: 12px; background: #fff3da; border-left: 4px solid #b67715; }
.muted { color: #4e5d70; } .back { font-size: .9rem; } :target { outline: 3px solid #7c9fdb; scroll-margin-top: 12px; }
@media (max-width: 650px) { body { padding: 10px; } section, article { padding: 12px; } .user, .assistant { margin-left: 0; margin-right: 0; } }
"""


def _text(value: object) -> str:
    if value is None:
        return "미기록"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _html(value: object) -> str:
    return escape(_text(value), quote=True)


def _public(value: object) -> object:
    if isinstance(value, dict):
        if value.get("type") in {"reasoning", "encrypted_reasoning"}:
            return "[비공개 추론 제외]"
        result = {}
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            if (
                normalized in _PRIVATE_FIELDS
                or normalized.endswith(
                    ("_api_key", "_private_key", "_password", "_token", "_secret")
                )
                or normalized.endswith(("_request", "_request_body"))
                or (normalized == "state" and isinstance(item, (dict, list)))
            ):
                result[key] = "[비공개 또는 중복 요청 자료 제외]"
            else:
                result[key] = (
                    _structured(item) if normalized in {"output", "arguments"} else _public(item)
                )
        return result
    if isinstance(value, list):
        return [
            _public(item)
            for item in value
            if not (
                isinstance(item, dict) and item.get("type") in {"reasoning", "encrypted_reasoning"}
            )
        ]
    if isinstance(value, str):
        prefix, payload = _notice_payload(value)
        if payload is None:
            decoded = _decode(value)
            if isinstance(decoded, (dict, list)):
                payload = decoded
                prefix = ""
        if payload is not None:
            value = prefix + json.dumps(_public(payload), ensure_ascii=False, sort_keys=True)
        value = re.sub(r"\b(?:sk|sess)-[A-Za-z0-9_-]{16,}", "[인증값 제외]", value)
        return re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [인증값 제외]", value)
    return value


def _decode(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def _notice_payload(value: object) -> tuple[str, dict | None]:
    if isinstance(value, str) and value.startswith("[app notice]") and "{" in value:
        prefix, _, suffix = value.partition("{")
        payload = _decode("{" + suffix)
        if isinstance(payload, dict):
            return prefix, payload
    return "", None


def _structured(value: object) -> object:
    return _public(_decode(value))


def _select(value: dict, fields: tuple[str, ...]) -> dict:
    return {key: value[key] for key in fields if key in value}


def _dump(value: object) -> str:
    return (
        "<pre>"
        + _html(json.dumps(_public(value), ensure_ascii=False, indent=2, sort_keys=True))
        + "</pre>"
    )


def _details(label: str, value: object) -> str:
    return "<details><summary>" + _html(label) + "</summary>" + _dump(value) + "</details>"


def _status(value: object) -> str:
    tone = (
        "good"
        if value in {"passed", "pass"}
        else (
            "bad"
            if value
            in {"failed", "fail", "blocked", "invalid", "invalid_judgment", "harness_error"}
            else ""
        )
    )
    label = _STATUS_LABELS.get(value, "상태") if isinstance(value, str) else "상태"
    return f'<span class="status {tone}">{_html(label)} · {_html(value)}</span>'


def _table(headers: tuple[str, ...], rows: list[list[str]]) -> str:
    if not rows:
        return '<p class="muted">저장된 항목 없음</p>'
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + "".join("<th>" + _html(header) + "</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + cell + "</td>" for cell in row) + "</tr>" for row in rows
        )
        + "</tbody></table></div>"
    )


def _campaign(report: dict) -> str | None:
    campaign = report.get("campaign")
    identifier = report.get("campaign_id")
    nested = campaign.get("id") if isinstance(campaign, dict) else campaign
    if identifier is not None and nested is not None and identifier != nested:
        raise ValueError("Conflicting campaign_id and campaign identity")
    value = identifier if identifier is not None else nested
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("Campaign identity must be a nonempty string")
    return value


def _identity(report: dict) -> dict:
    result = {field: report.get(field) for field in _IDENTITY_FIELDS}
    result["campaign_id"] = _campaign(report)
    result["campaign"] = report.get("campaign")
    context = report.get("relay_contexts")
    result["relay_context_sha256"] = (
        hashlib.sha256(
            json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if context is not None
        else None
    )
    return result


def _list(value: object, label: str) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be an array of objects")
    return value


def _identifier(item: dict, label: str, seen: set[str]) -> str:
    value = item.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {label} identity")
    if value in seen:
        raise ValueError(f"Duplicate or overlapping {label} identity: {value}")
    seen.add(value)
    return value


def _cases(report: dict) -> list[dict]:
    cases = _list(report.get("cases"), "cases")
    if "attempt" in report:
        attempt = report["attempt"]
        if not isinstance(attempt, dict):
            raise ValueError("attempt must be an object")
        scenario = attempt.get("scenario", {})
        cases = [
            *cases,
            {
                "id": scenario.get("id") or attempt.get("id"),
                "status": report.get("status"),
                "attempt": attempt,
            },
        ]
    return cases


def _runs(case: dict) -> list[dict]:
    if "attempt" in case:
        if "runs" in case or not isinstance(case["attempt"], dict):
            raise ValueError("A case must have runs or a single attempt, not both")
        return [{"status": case["attempt"].get("status"), "attempts": [case["attempt"]]}]
    return _list(case.get("runs", []), "runs")


def _validate(sources: list[dict]) -> None:
    if not sources:
        raise ValueError("At least one complete benchmark JSON report is required")
    expected = _identity(sources[0]["report"])
    if len(sources) > 1 and expected["campaign_id"] is None:
        raise ValueError("Multiple reports require an explicit matching campaign_id or campaign.id")
    case_ids: set[str] = set()
    run_ids: set[str] = set()
    attempt_ids: set[str] = set()
    for source in sources:
        report = source["report"]
        if report.get("format") != "puri-agent-simulator-1":
            raise ValueError("Only complete puri-agent-simulator-1 report objects are supported")
        identity = _identity(report)
        mismatches = [key for key in expected if expected[key] != identity[key]]
        if mismatches:
            raise ValueError("Campaign metadata mismatch: " + ", ".join(mismatches))
        for case in _cases(report):
            _identifier(case, "case", case_ids)
            for run in _runs(case):
                if "attempt" not in case:
                    _identifier(run, "run", run_ids)
                for attempt in _list(run.get("attempts", []), "attempts"):
                    _identifier(attempt, "attempt", attempt_ids)
                    _list(attempt.get("transcript", []), "transcript")
                    _list(attempt.get("actions", []), "actions")


def _coverage(report: dict) -> dict:
    cases = _cases(report)
    runs = [_runs(case) for case in cases]
    return {
        "listed_scenarios": len(cases),
        "saved_scenarios": sum(any(run.get("attempts") for run in group) for group in runs),
        "listed_runs": sum(len(group) for group in runs),
        "saved_runs": sum(bool(run.get("attempts")) for group in runs for run in group),
        "attempts": sum(len(run.get("attempts", [])) for group in runs for run in group),
    }


def _planned(sources: list[dict], key: str) -> int | None:
    values = [source["report"].get(key) for source in sources]
    return sum(values) if all(type(value) is int and value >= 0 for value in values) else None


def _brief(value: object, depth: int = 0) -> object:
    if isinstance(value, dict):
        items = [(key, item) for key, item in value.items() if item is not None]
        if depth >= 4:
            return {"saved_fields": [key for key, _ in items], "full_evidence": True}
        result = {key: _brief(item, depth + 1) for key, item in items[:10]}
        if len(items) > 10:
            result["additional_saved_fields"] = [key for key, _ in items[10:]]
        return result
    if isinstance(value, list):
        if depth >= 4:
            return {"saved_items": len(value), "full_evidence": True}
        result = [_brief(item, depth + 1) for item in value[:6]]
        if len(value) > 6:
            result.append({"additional_saved_items": len(value) - 6})
        return result
    if isinstance(value, str) and len(value) > 240:
        return value[:240] + "… [전체 증거 참조]"
    return value


def _result_summary(value: object) -> object:
    value = _structured(value)
    summary = _brief(value)
    if not isinstance(value, dict) or not isinstance(summary, dict):
        return summary
    receipt = value.get("receipt")
    if isinstance(receipt, dict) and isinstance(receipt.get("diagnosis"), dict):
        diagnosis = receipt["diagnosis"]
        candidates = []
        for candidate in diagnosis.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            item = _select(
                candidate,
                (
                    "id",
                    "current_route",
                    "input_host_api",
                    "input_device",
                    "opened",
                    "frames_received",
                    "frame_count",
                    "peak_level",
                    "speech_detected",
                    "failure_code",
                ),
            )
            peak = candidate.get("peak_level")
            if type(peak) in {int, float}:
                item["level"] = "zero" if peak == 0 else "nonzero"
            candidates.append(item)
        summary["receipt"] = {
            **_select(
                receipt,
                (
                    "operation_id",
                    "status",
                    "terminal",
                    "revision",
                    "recognition",
                    "settings_changed",
                ),
            ),
            "diagnosis": {
                **_select(
                    diagnosis,
                    ("state", "scope", "selected_candidate_id", "selection_basis", "failure_code"),
                ),
                "candidates": candidates,
            },
        }
    return summary


def _operation_id(value: object) -> object:
    if not isinstance(value, dict):
        return None
    receipt = value.get("receipt")
    return receipt.get("operation_id") if isinstance(receipt, dict) else value.get("operation_id")


def _unique_turn(values: list[object]) -> object:
    turns = {value for value in values if type(value) is int}
    return next(iter(turns)) if len(turns) == 1 else None


def _tool_entries(attempt: dict) -> list[dict]:
    entries = []
    by_call: dict[str, list[dict]] = {}
    anchors: dict[str, list[dict]] = {}
    observations = []
    events = attempt.get("events") or []
    calls = attempt.get("relay_calls") or attempt.get("M_invocations") or []
    for call in calls:
        response = call.get("response")
        if call.get("method", "respond") != "respond" or not isinstance(response, dict):
            continue
        for item in response.get("output") or []:
            if item.get("type") == "function_call" and isinstance(item.get("call_id"), str):
                anchors.setdefault(item["call_id"], []).append(call)
    for index, item in enumerate(attempt.get("archive", []), 1):
        if item.get("type") != "function_call":
            continue
        entry = {
            "kind": "model_tool",
            "turn": item.get("turn"),
            "tool": item.get("name"),
            "call": item,
            "results": [],
            "sources": [f"archive[{index}]"],
            "order": index,
        }
        entries.append(entry)
        if isinstance(item.get("call_id"), str):
            by_call.setdefault(item["call_id"], []).append(entry)

    def add_result(
        call_id: object,
        value: object,
        source: str,
        *,
        turn=None,
        tool=None,
        status=None,
        order=0,
        recorded=True,
    ):
        matches = by_call.get(call_id, []) if isinstance(call_id, str) else []
        if len(matches) == 1:
            entry = matches[0]
        else:
            entry = {
                "kind": "model_tool",
                "turn": turn,
                "tool": tool,
                "call": None,
                "call_id": call_id,
                "results": [],
                "sources": [],
                "order": order,
            }
            entries.append(entry)
            if not matches and isinstance(call_id, str):
                by_call[call_id] = [entry]
        entry["sources"].append(source)
        if not entry.get("tool"):
            entry["tool"] = tool
        entry.setdefault("turns", []).append(turn)
        if recorded and value is not None:
            decoded = _decode(value)
            existing = next(
                (result for result in entry["results"] if result["value"] == decoded), None
            )
            if existing is None:
                entry["results"].append({"value": decoded, "status": status})
            elif existing["status"] is None:
                existing["status"] = status
            observations.append((turn, decoded))

    archive = attempt.get("archive", [])
    for index, item in enumerate(archive, 1):
        if item.get("type") == "function_call_output":
            add_result(
                item.get("call_id"),
                item.get("output"),
                f"archive[{index}]",
                turn=item.get("turn"),
                order=index,
                recorded="output" in item,
            )
    for index, action in enumerate(attempt.get("actions", []), 1):
        result = action.get("result")
        if not isinstance(result, dict):
            continue
        for position, tool in enumerate(result.get("tool_results", []), 1):
            add_result(
                tool.get("call_id"),
                tool.get("result"),
                f"actions[{index}].tool_results[{position}]",
                turn=action.get("turn"),
                tool=tool.get("tool"),
                status=tool.get("status"),
                order=len(archive) + index,
                recorded="result" in tool,
            )
        if action.get("action") != "speak":
            observations.append((action.get("turn"), result))
    for entry in entries:
        call = entry["call"] or {}
        call_id = call.get("call_id", entry.get("call_id"))
        linked = anchors.get(call_id, []) if isinstance(call_id, str) else []
        entry["turn"] = _unique_turn(
            [entry["turn"], *entry.get("turns", []), *[item.get("turn") for item in linked]]
        )
        entry["call_id"] = call_id
        entry["relay_ids"] = [item["id"] for item in linked if "id" in item]
        entry["origins"] = list(
            dict.fromkeys(item["origin"] for item in linked if isinstance(item.get("origin"), str))
        )
        observations.extend((entry["turn"], result["value"]) for result in entry["results"])
    notices = [
        message for message in attempt.get("transcript", []) if message.get("role") == "notice"
    ]
    for event in [*events, *notices]:
        _, payload = _notice_payload(event.get("text"))
        observed = event.get("result") or (payload or {}).get("result")
        if isinstance(observed, dict):
            observations.append((event.get("turn"), observed))
    for index, item in enumerate(archive, 1):
        if item.get("type") != "app_operation":
            continue
        result = item.get("result", item.get("receipt"))
        operation_id = _operation_id(item) or _operation_id(result)
        exact = [turn for turn, observed in observations if observed == result]
        operation_turns = [
            turn
            for turn, observed in observations
            if operation_id is not None and _operation_id(observed) == operation_id
        ]
        turn = item.get("turn")
        if type(turn) is not int:
            turn = (
                _unique_turn(exact)
                if any(type(value) is int for value in exact)
                else _unique_turn(operation_turns)
            )
        entries.append(
            {
                "kind": "app_operation",
                "turn": turn,
                "tool": item.get("tool"),
                "record": item,
                "results": [{"value": result, "status": None}] if result is not None else [],
                "sources": [f"archive[{index}]"],
                "order": index,
            }
        )
    for index, action in enumerate(attempt.get("actions", []), 1):
        if action.get("action") == "speak":
            continue
        entries.append(
            {
                "kind": "user_action",
                "turn": action.get("turn"),
                "tool": action.get("action"),
                "record": action,
                "results": (
                    [{"value": action["result"], "status": None}]
                    if action.get("result") is not None
                    else []
                ),
                "sources": [f"actions[{index}]"],
                "order": index,
            }
        )
    notices = set()
    for index, event in enumerate(events, 1):
        if event.get("kind") == "decision":
            continue
        kind = "app_notice" if event.get("kind") == "app_notice" else "app_event"
        if kind == "app_notice":
            notices.add((event.get("turn"), event.get("text")))
        _, payload = _notice_payload(event.get("text"))
        record = {**event, **(payload or {})}
        entries.append(
            {
                "kind": kind,
                "turn": event.get("turn"),
                "tool": event.get("kind"),
                "record": record,
                "results": (
                    [{"value": record["result"], "status": None}]
                    if record.get("result") is not None
                    else []
                ),
                "sources": [f"events[{index}]"],
                "order": index,
            }
        )
    for index, message in enumerate(attempt.get("transcript", []), 1):
        if message.get("role") != "notice" or (message.get("turn"), message.get("text")) in notices:
            continue
        _, payload = _notice_payload(message.get("text"))
        record = payload or {}
        entries.append(
            {
                "kind": "app_notice",
                "turn": message.get("turn"),
                "tool": "app_notice",
                "record": record,
                "results": (
                    [{"value": record["result"], "status": None}]
                    if record.get("result") is not None
                    else []
                ),
                "sources": [f"transcript[{index}]"],
                "order": len(events) + index,
            }
        )
    ranks = {"user_action": 0, "model_tool": 1, "app_operation": 1, "app_notice": 2, "app_event": 2}
    return sorted(
        entries,
        key=lambda item: (
            type(item["turn"]) is not int,
            item["turn"] if type(item["turn"]) is int else 0,
            ranks[item["kind"]],
            item["order"],
        ),
    )


def _tool_ledger(attempt: dict) -> str:
    parts = [
        '<section data-kind="tool-ledger"><h4>턴별 도구 · 앱 동작 원장</h4>',
        '<p class="warning" data-kind="fixture-provenance" data-provenance="synthetic_fixture">'
        "시뮬레이터의 앱·오디오·브라우저 관측은 합성 fixture입니다. controlled=false 또는 "
        "fresh_dialogue는 실제 장치·브라우저 실측을 뜻하지 않습니다. 마이크 프레임 수신, 신호 레벨, "
        "VAD 음성 판정은 인식 성공과 다르며 recognition=not_exercised는 인식을 실행하지 않았다는 뜻입니다.</p>",
        '<p class="muted">턴은 저장된 call_id·결과·operation_id로만 연결합니다. 턴 안에서는 사용자 조작, '
        "도구/앱 보관 순서, 이벤트 순서로 나누어 표시하며 서로 다른 원본 사이의 시각 순서는 추정하지 않습니다. "
        "호출 완료는 설정 적용이 아니며 승인 대기는 그대로 보존합니다. 전체 자료는 아래 증거에서 펼칠 수 있습니다.</p>",
    ]
    entries = _tool_entries(attempt)
    current_turn = None
    opened = False
    labels = {
        "model_tool": "모델 도구 호출",
        "app_operation": "앱 작업 · 모델 호출 아님",
        "user_action": "사용자 조작 · 모델 호출 아님",
        "app_notice": "비동기 앱 알림 · 모델 호출 아님",
        "app_event": "앱 이벤트 · 모델 호출 아님",
    }
    for entry in entries:
        if not opened or entry["turn"] != current_turn:
            if opened:
                parts.append("</ol></section>")
            current_turn, opened = entry["turn"], True
            parts.append(
                f'<section data-kind="tool-turn" data-turn="{_html(current_turn)}"'
                f' data-turn-recorded="{str(type(current_turn) is int).lower()}"><h4>턴 {_html(current_turn)}</h4><ol>'
            )
        record = entry.get("record") or entry.get("call") or {}
        results = entry["results"]
        statuses = []
        for result in results:
            value = result["value"]
            status = value.get("status") if isinstance(value, dict) else None
            receipt = value.get("receipt") if isinstance(value, dict) else None
            if not isinstance(status, str) and isinstance(receipt, dict):
                status = receipt.get("status")
            if not isinstance(status, str):
                status = result["status"] if isinstance(result["status"], str) else "unknown"
            statuses.append(status)
        status = statuses[0] if len(set(statuses)) == 1 else "unknown"
        if not results:
            status = "missing_result" if entry["kind"] == "model_tool" else "unknown"
        identifiers = {
            "sources": entry["sources"],
            **_select(record, ("action_id", "phase", "card_id")),
        }
        if entry["kind"] == "model_tool":
            identifiers.update(
                call_id=entry.get("call_id"),
                relay_call_ids=entry.get("relay_ids", []),
                origins=entry.get("origins", []),
                invocation_status=record.get("status"),
            )
        parts.append(
            f'<li data-kind="tool-entry" data-entry-kind="{entry["kind"]}" data-status="{_html(status)}"'
            f' data-call-recorded="{str(entry.get("call") is not None).lower()}"'
            f' data-result-recorded="{str(bool(results)).lower()}">'
            f'<strong>{labels[entry["kind"]]} · {_html(_public(entry["tool"]))}</strong> · '
            + _status(status)
            + '<details data-kind="tool-details"><summary>인수 · 결과 · 연결 근거 보기</summary>'
            + '<div data-kind="tool-identifiers">'
            + _dump(identifiers)
            + "</div>"
        )
        if entry["kind"] == "model_tool":
            if entry.get("call") is None:
                parts.append(
                    '<p data-missing="call">함수 호출 기록 없음 · 결과와 인수의 연결을 추정하지 않음</p>'
                )
            arguments = _brief(_structured(record["arguments"])) if "arguments" in record else None
        else:
            arguments = _brief(_public(_select(record, ("changes", "target", "provider", "text"))))
        parts.append(
            '<div data-kind="tool-arguments"><strong>인수 / 조작</strong>'
            + (
                _dump(arguments)
                if arguments is not None
                else '<p data-missing="arguments">호출 인수 미기록</p>'
            )
            + "</div>"
        )
        parts.append(
            '<div data-kind="tool-result"><strong>기록된 결과 요약</strong>'
            + "".join(_dump(_result_summary(result["value"])) for result in results)
            + (
                '<p data-missing="result">결과 미기록 · 완료/적용을 추정하지 않음</p>'
                if not results
                else ""
            )
            + "</div></details></li>"
        )
    if entries:
        parts.append("</ol></section>")
    else:
        parts.append('<p data-missing="tools">저장된 도구/앱 동작 없음</p>')
    parts.append("</section>")
    return "".join(parts)


def _dialogue(attempt: dict) -> str:
    transcript = attempt.get("transcript", [])
    actions = attempt.get("actions", [])
    pending = [
        (index, action) for index, action in enumerate(actions) if action.get("action") != "speak"
    ]
    emitted: set[int] = set()
    parts = ['<section data-kind="dialogue"><h4>대화와 사용자 조작 · 저장 순서</h4>']
    for position, message in enumerate(transcript, 1):
        for index, action in pending:
            action_turn, message_turn = action.get("turn"), message.get("turn")
            reached = action_turn == message_turn or (
                type(action_turn) is int
                and type(message_turn) is int
                and action_turn < message_turn
            )
            if index not in emitted and reached:
                parts.append(_control(action))
                emitted.add(index)
        role = message.get("role")
        kind = role if role in {"user", "assistant", "notice"} else "other"
        speaker = {"user": "사용자 발화", "assistant": "Puri", "notice": "앱 알림 · 발화 아님"}.get(
            role, "기타 기록"
        )
        text = '<div class="literal">' + _html(_public(message.get("text"))) + "</div>"
        if role == "notice":
            text = "<details><summary>앱 알림 원문 보기</summary>" + text + "</details>"
        parts.append(
            f'<div class="message {kind}" data-role="{kind}" data-sequence="{position}">'
            f'<strong>{speaker}</strong> · 턴 {_html(message.get("turn"))} · 기록 {position}'
            + text
            + "</div>"
        )
    for index, action in pending:
        if index not in emitted:
            parts.append(_control(action, unmatched=True))
    if not transcript:
        parts.append(
            '<p class="warning">저장된 대화 전사 없음. 아래 동작/오류 기록을 확인하세요.</p>'
        )
    parts.append("</section>")
    return "".join(parts)


def _control(action: dict, *, unmatched: bool = False) -> str:
    label = "전사 연결 없음 · " if unmatched else ""
    return (
        '<div class="message control" data-role="control"><strong>'
        + label
        + "사용자 조작 · 발화 아님</strong>"
        + (
            f' · 턴 {_html(action.get("turn"))} · {_html(action.get("action"))}'
            f' · 대상 {_html(action.get("target"))} · 제공자 {_html(action.get("provider"))}'
            '<div class="literal">' + _html(_public(action.get("text", ""))) + "</div></div>"
        )
    )


def _judgment(attempt: dict, report: dict) -> str:
    judgment = attempt.get("judgment") or {}
    parts = ['<section data-kind="code-gates"><h4>코드 게이트 · 상태/금지/경로 조건</h4>']
    code = attempt.get("code")
    if isinstance(code, dict):
        parts.append(
            _table(
                ("코드 조건", "기록된 값"),
                [
                    [label, _html(_public(code[key]))]
                    for key, label in (
                        ("accepted_state", "목표 상태 도달"),
                        ("prohibitions_pass", "금지 행동 없음"),
                    )
                    if key in code
                ],
            )
        )
        parts.append(
            _details(
                "코드 게이트 상세 JSON",
                {key: value for key, value in code.items() if key != "state"},
            )
        )
        if "state" in code:
            parts.append('<p class="muted">전체 최종 상태 스냅샷은 원본 JSON에 보존됩니다.</p>')
    else:
        parts.append("<p>코드 게이트 미기록</p>")
    parts.append(
        '</section><section data-kind="sol-rubric"><h4>Sol 루브릭 · 코드 게이트와 별도</h4>'
    )
    parts.append(_status(judgment.get("status")))
    parts.append(
        _details(
            "심사 모델 · 버전 · 오류 상세",
            _select(
                judgment,
                (
                    "model",
                    "effort",
                    "revision",
                    "rejudgments",
                    "code",
                    "errors",
                    "error",
                    "status",
                ),
            ),
        )
    )
    advisory = report.get("advisory", [])
    rows = []
    for criterion in judgment.get("criteria", []):
        name = criterion.get("name")
        rows.append(
            [
                _html(name),
                "권고" if name in advisory else "필수/조건부 · 원본 적용 조건 참조",
                _status(criterion.get("verdict")),
                _html(criterion.get("turn")),
                '<div class="literal">' + _html(_public(criterion.get("quote"))) + "</div>",
            ]
        )
    parts.append(_table(("기준", "구분", "판정", "턴", "인용"), rows))
    parts.append(
        '</section><section data-kind="d-actual"><h4>D 실제 분류/적용값 · Sol 판정 아님</h4>'
    )
    calls = attempt.get("D_invocations")
    if calls is None:
        calls = [
            call for call in attempt.get("relay_calls", []) if call.get("method") == "classify"
        ]
    for call in calls:
        actual = _select(
            call,
            (
                "id",
                "turn",
                "origin",
                "status",
                "code",
                "error",
                "effective_route",
                "effective_effort",
                "source",
                "fallback_code",
                "decision",
                "upstream_context_id",
            ),
        )
        actual["response"] = _public(call.get("response"))
        parts.append(_details(f'D 호출 {call.get("id")} · {call.get("status", "미기록")}', actual))
    if not calls:
        parts.append("<p>D 호출 미기록. 알림 경로의 우회는 D 호출로 세지 않습니다.</p>")
    parts.append(
        '</section><section data-kind="d-judgment"><h4>D에 대한 Sol 경로/effort 판정 · 실제 분류와 별도</h4>'
    )
    parts.append(
        _table(
            ("D 호출 ID", "경로 판정", "effort 판정", "인용"),
            [
                [
                    _html(row.get("call_id")),
                    _status(row.get("route")),
                    _status(row.get("effort")),
                    '<div class="literal">' + _html(_public(row.get("quote"))) + "</div>",
                ]
                for row in judgment.get("routing", [])
            ],
        )
    )
    parts.append("</section>")
    return "".join(parts)


def _evidence(attempt: dict) -> str:
    parts = [
        '<details data-kind="action-evidence"><summary>동작 · 도구 · 호출 증거 펼치기</summary>'
    ]
    transcript_text = {message.get("text") for message in attempt.get("transcript", [])}
    shown_outputs: dict[str, list[object]] = {}
    for index, action in enumerate(attempt.get("actions", []), 1):
        item = _select(
            action, ("turn", "action", "text", "target", "provider", "result", "error", "status")
        )
        result = item.get("result")
        if isinstance(result, dict):
            for tool in result.get("tool_results", []):
                if isinstance(tool.get("call_id"), str) and "result" in tool:
                    shown_outputs.setdefault(tool["call_id"], []).append(_decode(tool["result"]))
            item["result"] = {
                key: value
                for key, value in result.items()
                if key != "text" or value not in transcript_text
            }
        kind_label = "사용자 발화" if action.get("action") == "speak" else "사용자 조작 · 발화 아님"
        parts.append(_details(f'{kind_label} {index} · {action.get("action", "미기록")}', item))
    for index, item in enumerate(attempt.get("archive", []), 1):
        kind = item.get("type")
        if kind not in {"function_call", "function_call_output", "app_operation", "report_offer"}:
            continue
        if kind == "function_call_output" and "output" in item:
            call_id = item.get("call_id")
            if isinstance(call_id, str) and _decode(item["output"]) in shown_outputs.get(
                call_id, []
            ):
                continue
        public = {
            key: _structured(value) if key in {"arguments", "output"} else value
            for key, value in item.items()
        }
        parts.append(_details(f"도구/앱 보관 기록 {index} · {kind}", public))
    calls = attempt.get("relay_calls", [])
    if not calls:
        calls = [*attempt.get("D_invocations", []), *attempt.get("M_invocations", [])]
    parts.append(
        _details(
            "D/M/지원 전송 호출 상태 · 요청 본문 제외",
            [
                _select(
                    call,
                    (
                        "id",
                        "turn",
                        "method",
                        "origin",
                        "status",
                        "code",
                        "error",
                        "upstream_context_id",
                    ),
                )
                for call in calls
            ],
        )
    )
    for index, call in enumerate(attempt.get("model_calls", []), 1):
        metadata = _select(
            call,
            (
                "role",
                "model",
                "requested_model",
                "returned_model",
                "effort",
                "status",
                "terminal_status",
                "code",
                "error",
                "rejudgment",
                "access_path",
                "usage",
                "cost_usd",
                "cost_basis",
            ),
        )
        if "value" in call:
            metadata["structured_value"] = _structured(call["value"])
        parts.append(
            _details(
                f'Terra/Sol 호출 {index} · {call.get("role", "미기록")} · 각 심사 시도 보존',
                metadata,
            )
        )
    for field, label in (
        ("events", "앱 이벤트"),
        ("effective_decisions", "적용 결정 · 알림 우회 포함"),
        ("access_attempts", "모델 접근 시도 · 요청/인증 제외"),
        ("provider_signup", "합성 가입 동작"),
        ("injection_device_observations", "주입 노출 관측"),
    ):
        if field in attempt:
            parts.append(_details(label, attempt[field]))
    parts.append("</details>")
    return "".join(parts)


def _render(sources: list[dict]) -> tuple[str, dict]:
    coverage = Counter()
    for source in sources:
        coverage.update(_coverage(source["report"]))
    coverage["planned_scenarios"] = _planned(sources, "planned_scenarios")
    coverage["planned_runs"] = _planned(sources, "planned_runs")
    parts = [
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">",
        "<title>Puri 최신 캠페인 대화 보고서</title><style>",
        _STYLE,
        '</style></head><body id="top">',
        "<h1>Puri 최신 캠페인 대화 보고서</h1>",
        "<p>지정한 한 캠페인의 저장된 보고서만 표시합니다. 과거 결과를 자동 검색하거나 최신 성공으로 덮어쓰지 않습니다.</p>",
        '<p class="warning">오프라인 표시 전용입니다. 상태·판정을 재계산하지 않으며, 미저장 실행은 성공이 아닙니다. '
        "제어 smoke는 실측 통과가 아닙니다. 비공개 추론/암호화 추론/모델 요청 본문/인증 필드와 반복 상태 스냅샷은 내보내지 않습니다.</p>",
    ]
    identity = _identity(sources[0]["report"])
    if identity["M_model"] is not None:
        parts.append('<section data-kind="provider-identity"><h2>측정 모델</h2>')
        parts.append(
            _table(
                ("역할", "공급자", "모델"),
                [
                    ["Puri 응답 M", _html(identity["M_provider"]), _html(identity["M_model"])],
                    ["분류 D", _html(identity["D_provider"]), _html(identity["D_model"])],
                ],
            )
        )
        parts.append(
            "<p>원본 요청의 모델 표기는 실제 공급자 구성과 다를 수 있습니다. "
            "실험 연결 정보는 캠페인 식별 상세에 보존합니다.</p></section>"
        )
    if identity["campaign_id"] is None:
        parts.append(
            '<p class="warning">캠페인 ID 미기록: 단일 레거시 원본만 표시하며 동일 캠페인 여부는 확인되지 않았습니다.</p>'
        )
    parts.append(
        '<section data-kind="campaign"><h2>캠페인 · 빌드 · 심사 식별</h2>'
        + _details("식별 정보 펼치기", identity)
        + "</section>"
    )
    parts.append('<section data-kind="coverage"><h2>범위 · 시나리오 / 논리 실행 / 시도</h2>')
    parts.append(
        _table(
            ("항목", "계획", "원본에 열거", "시도 저장 있음"),
            [
                [
                    "시나리오",
                    _html(coverage["planned_scenarios"]),
                    str(coverage["listed_scenarios"]),
                    str(coverage["saved_scenarios"]),
                ],
                [
                    "논리 실행",
                    _html(coverage["planned_runs"]),
                    str(coverage["listed_runs"]),
                    str(coverage["saved_runs"]),
                ],
                [
                    "시도 (attempt)",
                    "계획값과 별개",
                    str(coverage["attempts"]),
                    str(coverage["attempts"]),
                ],
            ],
        )
    )
    parts.append(
        "<p>저장 수는 실제 attempt가 있는 항목을 셉니다. 차단되어 시도가 없는 실행은 열거 수에만 포함됩니다. "
        "계획 수는 원본별 계획의 합이며, 누락된 계획은 추정하지 않습니다.</p></section>"
    )
    parts.append('<section data-kind="sources"><h2>원본 파일 · SHA-256 · 저장 상태</h2>')
    for source in sources:
        report = source["report"]
        parts.append("<h3>" + _html(source["name"]) + "</h3>" + _status(report.get("status")))
        parts.append(
            _details(
                "원본 해시 · 저장 상태 상세",
                {
                    "filename": source["name"],
                    "sha256": source["sha256"],
                    **_select(
                        report,
                        (
                            "created_at",
                            "status",
                            "scenario_status",
                            "controlled_smoke",
                            "live_pass",
                            "selection",
                            "planned_scenarios",
                            "saved_scenarios",
                            "planned_runs",
                            "saved_runs",
                            "missing_prerequisites",
                            "error",
                            "code",
                            "access_failure",
                            "limits_note",
                            "calibration",
                            "summary",
                        ),
                    ),
                },
            )
        )
    parts.append('</section><nav aria-label="사례와 실행 탐색"><h2>사례 / 실행 바로가기</h2><ul>')
    cases = [(source, case) for source in sources for case in _cases(source["report"])]
    for case_index, (_, case) in enumerate(cases, 1):
        parts.append(
            f'<li><a href="#case-{case_index}">{_html(case["id"])}</a> · '
            + _status(case.get("status"))
            + "<ul>"
        )
        for run_index, run in enumerate(_runs(case), 1):
            parts.append(
                f'<li><a href="#case-{case_index}-run-{run_index}">{_html(run.get("id", "단일 시도 형식"))}</a> · '
                + _status(run.get("status"))
                + "</li>"
            )
        parts.append("</ul></li>")
    parts.append("</ul></nav>")
    for case_index, (source, case) in enumerate(cases, 1):
        parts.append(
            f'<article id="case-{case_index}" data-kind="case" data-id="{_html(case["id"])}" data-status="{_html(case.get("status"))}"><h2>{_html(case["id"])}</h2>'
            + _status(case.get("status"))
        )
        parts.append(
            _details(
                "시나리오 집계 상세",
                {
                    "source": source["name"],
                    **_select(
                        case,
                        (
                            "category",
                            "split",
                            "successes",
                            "logical_runs",
                            "advisory_failures",
                            "missing_prerequisites",
                            "error",
                        ),
                    ),
                },
            )
        )
        for run_index, run in enumerate(_runs(case), 1):
            parts.append(
                f'<article id="case-{case_index}-run-{run_index}" data-kind="run" data-id="{_html(run.get("id"))}" data-status="{_html(run.get("status"))}"><h3>논리 실행 · {_html(run.get("id", "ID 미기록 · 단일 시도 형식"))}</h3>'
                + _status(run.get("status"))
            )
            run_metadata = _select(run, ("error", "code", "missing_prerequisites", "harness_error"))
            if run_metadata:
                parts.append(_details("실행 오류 상세", run_metadata))
            attempts = run.get("attempts", [])
            if not attempts:
                parts.append(
                    '<p class="warning">저장된 시도 없음 · 실행 완료/성공을 뜻하지 않습니다.</p>'
                )
            for attempt in attempts:
                parts.append(
                    '<article data-kind="attempt" data-id="'
                    + _html(attempt["id"])
                    + '" data-status="'
                    + _html(attempt.get("status"))
                    + '"><h4>시도 · '
                    + _html(attempt["id"])
                    + "</h4>"
                    + _status(attempt.get("status"))
                )
                metadata = _select(
                    attempt,
                    (
                        "started_at",
                        "finished_at",
                        "controlled",
                        "error",
                        "access_failure",
                        "harness_error",
                        "invalid_user_action",
                        "missing_prerequisites",
                        "counts",
                        "cost",
                        "measurement_origin",
                    ),
                )
                if "code" in attempt and not isinstance(attempt["code"], dict):
                    metadata["code"] = attempt["code"]
                if attempt.get("error"):
                    parts.append('<p class="warning">' + _html(_public(attempt["error"])) + "</p>")
                parts.append(_details("시도 식별 · 호출량 · 오류 증거", metadata))
                parts.append(_dialogue(attempt))
                parts.append(_tool_ledger(attempt))
                parts.append(_judgment(attempt, source["report"]))
                parts.append(_evidence(attempt))
                parts.append("</article>")
            parts.append('<a class="back" href="#top">맨 위 / 사례 탐색</a></article>')
        parts.append("</article>")
    parts.append("</body></html>\n")
    return "".join(parts), dict(coverage)


def render_html(reports: list[str | Path], output: str | Path) -> dict:
    sources = []
    destination = Path(output)
    for filename in reports:
        path = Path(filename)
        if destination.resolve() == path.resolve() or (
            destination.exists() and path.exists() and destination.samefile(path)
        ):
            raise ValueError("HTML output must not overwrite an input report")
        raw = path.read_bytes()
        report = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(report, dict):
            raise ValueError("Each input must be a complete JSON report object")
        sources.append(
            {"name": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "report": report}
        )
    _validate(sources)
    document, coverage = _render(sources)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(document.encode("utf-8"))
    return {
        "status": "rendered",
        "output": str(destination),
        "sources": len(sources),
        "coverage": coverage,
    }
