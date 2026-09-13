from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.metrics import AMI_WORDS, ROLES, load_ami_words
HERE = Path(__file__).resolve().parent
RETAINED = HERE.parent / "retained"
POLICY_BUNDLE = RETAINED / "historical_policy_inputs.jsonl.gz"
TRANSLATION_BUNDLE = RETAINED / "translated_texts.jsonl.gz"
BASELINE = "e802023f2f3b25762a45684e7e136a956d976dfa"
MODEL = "google/gemma-4-26b-a4b-it"
MAX_TOKENS = 512
PROMPT = """# Role: Meaning-preserving Korean batch translator
Translate the complete English parent into natural Korean while preserving every positional segment.

The input is JSON. `parent_text` is the complete parent and `segments` partitions that exact text in order. Read the whole parent before translating any segment. Segment IDs are positional correspondence labels only: they are not person identities, speaker names, or evidence that a segment is pure single-speaker speech.

Return JSON only with schema `meaning-first-e1-output-1` and an ordered `translations` array. Return exactly one object per input segment, in the same order, with exactly its `id` and a Korean `text` string. Do not merge, omit, duplicate, or reorder IDs.

Preserve conversational meaning, question/answer relations, agreement, negation, correction, uncertainty, and claim attribution. Do not invent names, roles, relationships, facts, or speaker identities. Do not invert negation or agreement. Do not rewrite multiple turns as one monologue. Keep incomplete or noisy ASR meaning incomplete when it cannot be recovered from the shared parent. Translate only the supplied source text."""
PROVIDER = {
    "model": MODEL,
    "reasoning": {"effort": "none"},
    "temperature": 0.6,
    "provider": {
        "order": ["wafer", "cloudflare", "deepinfra"],
        "only": ["wafer", "cloudflare", "deepinfra"],
        "allow_fallbacks": True,
    },
    "max_tokens": MAX_TOKENS,
}
HISTORICAL_PROVIDER = {**PROVIDER, "max_tokens": 100}

ASSESSMENTS = {
    "33c69f5e-bece-4c31-bec7-37d2fb42565b": ("equal", "R0는 한 흐름으로 자연스럽고 R2는 ‘that.’을 ‘그거요.’, ‘And’를 ‘그리고’로 고립시켰지만, 순서대로 합친 내용에서 명확한 사실·부정·화행 반전은 확인되지 않는다.", "R2의 단편화는 명백한 가독성 저하이지만 이 기록만으로 주 의미 손상을 입증하지 못한다.", "broken_fragment"),
    "147e5b68-0232-4e76-a587-8ddfbb2b817a": ("worse", "R0는 ‘Bye. You.’를 ‘안녕히 계세요. 당신도요.’로 연결했지만 R2는 ‘You. Alright.’를 ‘너 말이야. 알았어.’로 바꿨다.", "짧은 응답 자체가 문제가 아니라 잘못 묶인 ‘You.’의 화행이 상대 인사에서 지목으로 변했다.", "broken_fragment"),
    "8ee82188-f635-48d8-8502-352b673851ec": ("worse", "R0의 ‘이제부터는 직접 하시는 게’와 달리 R2는 ‘take it from’만 보고 ‘그렇게 가져가는 게’로 옮기고 ‘here.’를 ‘여기요.’로 분리했다.", "구절 경계가 관용적 ‘take it from here’의 의미를 잃게 했다.", "broken_fragment"),
    "9a75496a-26b7-42f8-a41b-f6ed81d41dcc": ("worse", "‘by two’는 문맥상 곱셈·나눗셈이 모호하므로 두 번역의 연산 선택에는 우열을 부여하지 않는다. 결정적 차이는 R2가 ‘thirty five’를 ‘그럼 30이네.’와 ‘다섯.’으로 갈라 35라는 수를 잃은 점이다.", "연산 해석이 아니라 숫자 35의 합성 손실 때문에 R2가 더 나쁘다; R0 역시 완전하다고 판정하지 않는다.", "broken_fragment"),
    "eb86a37a-23b6-4054-a9aa-3d5b38e962a3": ("worse", "R0 ‘태양광 발전은 확실히 그렇겠지만, 제 생각에는…’가 보존한 절을 R2는 ‘그… / 태양광… / 근데 저… / 생각 중…’으로 쪼갰다.", "동일 화자 절의 문법 관계가 끊겼다.", "broken_fragment"),
    "5f8dedf2-a8e8-4520-8df4-36f5a4eeee63": ("unjudgeable", "R0 ‘진짜 자연스럽네요’와 R2 ‘유기농인가요, / 진짜요.’는 모호한 ‘Organic, really.’를 서로 다르게 해석한다.", "GT가 뒤 구간을 혼합으로 표시하고 원문도 모호하여 의미 우열은 판단 불가지만 R2 가독성은 단편화로 낮다.", "broken_fragment"),
    "8766ce13-7ef6-4f8b-a09a-86cd0883a603": ("worse", "R0 ‘그 슬라이드가 그랬다고요? … 전 이거 하나도 안 바꿨는데요.’가 R2의 ‘이 슬라이드 / 방금 그거 슬라이드였어요? / 였어요 / 음… / 저거요?’보다 명확하다.", "질문과 부정 진술 사이의 핵심 의미는 남지만 앞 질문이 여러 불완전 조각으로 붕괴했다.", "broken_fragment"),
    "5c333233-1622-4850-a60e-ca8811613108": ("worse", "R0는 ‘기술을 보는 게 아니라 기능을 먼저’라고 보존했지만 R2는 ‘at the function.’을 ‘행사에서요.’로 번역했다.", "문맥을 잃은 function의 다의어 오역과 다수 단편이 핵심 대조를 약화했다.", "broken_fragment"),
    "5b09905d-fcc2-4c75-b03e-316edfe2df95": ("worse", "R0 ‘아무것도 버리지 않고 있어요’와 달리 R2는 ‘not / actually throwing / anything away’를 ‘아니요 / 진짜 던지네 / 뭐든 멀리요’로 만들었다.", "부정 범위와 ‘throw away=버리다’ 의미가 실제로 뒤집히거나 소실되었다.", "broken_fragment"),
    "4a8dca47-d874-420b-a30d-60dd9184cfde": ("worse", "R0 ‘작업 분할’이 R2의 고립된 ‘breakdown.’에서 ‘무너졌어’가 됐다.", "독립 번역 문맥 손실로 전문 용어 의미가 바뀌었다.", "broken_fragment"),
    "9eda2eee-8143-42d9-beef-8dc31af891e8": ("equal", "R0는 조건문을 자연스럽게 보존하고 R2는 ‘You’를 단독 ‘너’로 노출하지만, 뒤 번역은 원문의 큰 조건 의미를 보존한다.", "고립된 호칭은 가독성과 자연스러움을 해치지만 명확한 주 의미 오류로 계산하지 않는다.", "broken_fragment"),
    "9d25928d-e12d-4455-b608-66145abbb969": ("worse", "R0 ‘그건 그건 저희가 결정할…’이 R2에서 ‘그건 / 그건 / 결정 하나요 / 우리’로 붕괴했다.", "미완 문장이라는 원문 한계를 넘어 술어-주어 관계까지 분리되었다.", "broken_fragment"),
    "d928e75b-4caf-4368-a614-b230a84ab69b": ("worse", "R0는 ‘태그를 달 수 있을까요?’라는 질문을 보존하지만 R2의 본절은 ‘태그를 달 수 있을 것 같아요.’라는 추정 진술이고, ‘Can’을 별도 ‘할 수 있어?’로 떼었다.", "질문 양태가 본절에서 추정 진술로 바뀌어 주 화행이 약화되거나 변했다; 오염 47→0은 이 의미 손상을 상쇄하지 않는다.", "broken_fragment"),
    "f3c5a2b5-fee1-47a8-953b-e1f80e8094a4": ("worse", "R0는 ‘지금 거기 있는 … 그걸 뭘로 바꿔야’라는 연결을 보존하지만 R2는 ‘at’를 ‘어디요?’, ‘to’를 ‘~로’로 독립 번역했다.", "전치사와 목적어 분리로 여러 실제 의미 불일치가 생겼다.", "broken_fragment"),
    "a7dbe963-1731-4bb3-ba81-0377eab8b6a8": ("worse", "R0 ‘똑같이 생긴 게 여러 개’와 달리 R2는 ‘있는 것보다 더 많다 / 여러 개 / 저거 다 똑같아’로 흩어진다.", "ASR 자기수정을 한 문장으로 회복할 문맥이 분할로 사라졌다.", "broken_fragment"),
    "574c2f12-d8e7-49cd-957d-e1eec876671e": ("worse", "R0는 ‘로봇이 어떻게 설정… 만약에 5, 5 그린…’을 유지하지만 R2는 ‘So if five’를 ‘그래서 만약 다섯 명이면’으로 해석했다.", "뒤의 ‘five green’ 문맥을 못 본 독립 번역이 사람 수를 발명했다.", "broken_fragment"),
    "2b2a9153-b3cb-438b-b037-32fa5039b057": ("worse", "R0 ‘네덜란드 쪽은 안 그랬어요 … 아마 문서화 문제’가 R2에서 ‘네덜란드요 / 많이 / 안 했어요… / 문서화요’가 됐다.", "이미 깨진 ASR이지만 R2는 명사와 부정을 더 분리해 회복 가능성을 낮췄다.", "broken_fragment"),
    "3e085e5a-24ee-40e9-a5f2-74d368927071": ("equal", "R0 ‘우리가 생각 못 했던 거네요’가 더 자연스럽고 R2는 ‘우리가 생각지도 못한 거요 / ~에 대해서요’로 끊겼지만 후반의 동의와 합리적이라는 판단을 포함한 큰 뜻은 비슷하다.", "전치사 단편은 가독성 손실이지만 이 기록에서 별도의 명확한 주 의미 오류는 확인하지 못한다.", "broken_fragment"),
    "b1ca9396-ddeb-4f9e-9e2d-f0d6dc93f182": ("worse", "R0 ‘확실히 알겠어요 … 진짜 좋네요’가 R2 ‘그건 / 확인. / 그게 그게 / 그거 좋네요’가 됐다.", "clear의 서술 의미가 명령/명사형 ‘확인’으로 바뀌고 반복 단편이 생겼다.", "broken_fragment"),
    "102972ae-4b2e-4055-ad0e-4f467b6b1ef7": ("worse", "R0는 Joe가 두어 시간을 낼 수 있고 그 정도 걸릴 수 있다는 조건을 잇지만 R2는 ‘is’를 ‘인가요?’로 만들고 ‘what it might take’를 ‘뭘 필요로’로 바꿨다.", "관계절 경계가 질문이 아닌 것을 질문으로 만들고 소요시간 의미를 훼손했다.", "broken_fragment"),
}

E1_SPECS = [
    ("Q1", "target_question_answer", "0d657a93-2839-4f41-bb70-a3edddcf8857", [9], {"question": "Do you watch the new season?", "answer": "No.", "facts": ["The answer negates watching the new season, then asks whether it is obtained online or by downloading."], "recoverability": "high_despite_no_space_after_answer"}),
    ("Q2", "target_question_answer", "a35203d8-ae53-4bb5-a6ce-b4c6bb0531b7", [2], {"question": "The day?", "answer": "Yeah.", "facts": ["The answer is an affirmative response."], "recoverability": "high_from_accepted_text"}),
    ("Q3", "target_question_answer", "f38c34bc-0062-4a41-bb9a-e4e32b691021", [7], {"question": "we're in agreement on that?", "answer": "Unfortunately, I think we are.", "facts": ["The answer affirms agreement while marking reluctance."], "recoverability": "high_from_accepted_text"}),
    ("Q4", "target_question_answer", "c5682f84-f79d-46ea-9d13-a19217ea5f38", [6], {"question": "What what do you think, Craig?", "answer": "Well, did you not say it was the the adults? That were going for the the voice recognition?", "facts": ["The response challenges or corrects the premise by asking whether adults were the intended voice-recognition users."], "recoverability": "high_despite_ASR_joined_sentence_boundary"}),
    ("R1", "target_A_B_A", "9a75496a-26b7-42f8-a41b-f6ed81d41dcc", [7, 8], {"facts": ["First turn states a calculation result.", "Middle turn says ‘Right.’ and is an agreement response.", "The first role resumes with ‘So’."], "recoverability": "meaning_partly_limited_by_ASR_number_wording"}),
    ("R2", "target_A_B_A", "0294cb49-e5c7-4daa-9a44-839e6fde9b88", [9, 12], {"facts": ["First turn claims basic functionality is primary.", "Short middle turn includes ‘Yeah.’.", "The first role resumes with a menu-button proposal."], "recoverability": "middle_turn_is_short_and_ASR_noisy"}),
    ("C1", "target_A_B_C", "63954a9b-e069-41bc-821f-f82a248b0855", [9, 10], {"question": "You guys felt like there's enough teamwork and all.", "answers": ["Yeah.", "I think so."], "facts": ["Two distinct responding roles separately express agreement."], "recoverability": "high_from_accepted_text"}),
    ("C2", "target_A_B_C", "4bd08541-357d-4a3c-9a9a-637080b7a2aa", [11, 12], {"facts": ["Role C says the wire in the back might be loose.", "Role D responds ‘Okay.’.", "Role B then says ‘Yeah. You wanna’, an incomplete accepted-text continuation."], "checked_role_sequence": ["C", "D", "B"], "retained_guard_evidence": {"same_speaker_stratum": False, "span_verified_changes": 2, "failures": []}, "recoverability": "first_claim_and_two_response_onsets_clear; final_turn_incomplete"}),
    ("S1", "guard_same_speaker", "33238b0b-50d0-46bc-abe6-ad596463bb78", [], {"facts": ["One role says the group has not narrowed down the requirements."], "negation": "can't / haven't", "recoverability": "high_from_accepted_text"}),
    ("S2", "guard_same_speaker", "92fce92f-75da-4b85-9f8a-b4d99638be00", [], {"facts": ["One role describes the better solution as fixing this problem by means of the mono microphone socket."], "recoverability": "high_from_accepted_text"}),
    ("U1", "guard_timing_unknown", "8465ae6c-5f27-457f-bcc1-42944ff0294e", [], {"facts": ["The accepted text contains mutual negation: ‘That's not you’ and ‘That's not me’."], "recoverability": "boundary_not_frozen_due_unmapped_or_mixed_timing"}),
    ("O1", "guard_overlap", "3411dfa5-a013-49c7-8f77-19528eccb9c2", [], {"facts": ["Every accepted token overlaps GT speech from roles A and D."], "recoverability": "overlap_not_resolved_to_a_single_role"}),
]

CONTROL_TEXTS = ("Interesting.", "Cool.", "Yeah. Yeah.")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

SHORT_RESPONSE_MIXED_IDS = {
    "147e5b68-0232-4e76-a587-8ddfbb2b817a",
    "8ee82188-f635-48d8-8502-352b673851ec",
    "9a75496a-26b7-42f8-a41b-f6ed81d41dcc",
    "5f8dedf2-a8e8-4520-8df4-36f5a4eeee63",
    "8766ce13-7ef6-4f8b-a09a-86cd0883a603",
    "5c333233-1622-4850-a60e-ca8811613108",
    "5b09905d-fcc2-4c75-b03e-316edfe2df95",
    "9eda2eee-8143-42d9-beef-8dc31af891e8",
    "9d25928d-e12d-4455-b608-66145abbb969",
    "f3c5a2b5-fee1-47a8-953b-e1f80e8094a4",
    "a7dbe963-1731-4bb3-ba81-0377eab8b6a8",
    "574c2f12-d8e7-49cd-957d-e1eec876671e",
    "2b2a9153-b3cb-438b-b037-32fa5039b057",
    "3e085e5a-24ee-40e9-a5f2-74d368927071",
}
E0_LOW_CERTAINTY_FRAGMENT_IDS = {
    "33c69f5e-bece-4c31-bec7-37d2fb42565b",
    "9eda2eee-8143-42d9-beef-8dc31af891e8",
    "3e085e5a-24ee-40e9-a5f2-74d368927071",
}

def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_bundle(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def provider_body(logical_payload: dict[str, Any]) -> dict[str, Any]:
    body = {
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"<input>\n{canonical(logical_payload)}\n</input>"},
        ],
        **PROVIDER,
    }
    return body


def historical_body(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": request["system_prompt"]},
            {"role": "user", "content": f"<input>\n{request['text']}\n</input>"},
        ],
        **HISTORICAL_PROVIDER,
    }


def role_runs(parent: dict[str, Any]) -> list[dict[str, Any]]:
    attributed = {row["token_id"]: row for row in parent["r0"]["attribution"]}
    runs: list[dict[str, Any]] = []
    for token in parent["tokens"]:
        row = attributed[token["token_id"]]
        role = row["roles"][0] if row["status"] == "attributable" and len(row["roles"]) == 1 else row["status"].upper()
        if not runs or runs[-1]["gt_role_or_status"] != role:
            runs.append({"gt_role_or_status": role, "token_ids": [], "text": ""})
        runs[-1]["token_ids"].append(token["token_id"])
        runs[-1]["text"] += token["text"]
    return runs


def translation_outcome(child: dict[str, Any]) -> dict[str, Any]:
    actual = child.get("actual_acquisition")
    if actual is not None:
        return {"status": actual["status"], "response": actual.get("response"), "error": actual.get("error"), "original_request_id": child["original_request_id"]}
    return {"status": child.get("original_status"), "response": None, "error": "retained_actual_acquisition_unavailable", "original_request_id": child.get("original_request_id")}


def annotation_identity(meeting: str) -> list[dict[str, Any]]:
    result = []
    for role in ROLES:
        path = AMI_WORDS / f"{meeting}.{role}.words.xml"
        result.append({"role": role, "path": str(path), "exists": path.exists(), "sha256": digest_file(path) if path.exists() else None})
    return result


def boundary_witness(parent: dict[str, Any], meeting: str, position: int) -> dict[str, Any]:
    left_token = parent["tokens"][position - 1]
    right_token = parent["tokens"][position]
    left_attr = parent["r0"]["attribution"][position - 1]
    right_attr = parent["r0"]["attribution"][position]
    if left_token.get("source_end_sample") is None or right_token.get("source_start_sample") is None:
        raise RuntimeError(f"unmapped GOLD cut {parent['parent_id']}:{position}")
    if left_attr["status"] != "attributable" or right_attr["status"] != "attributable" or len(left_attr["roles"]) != 1 or len(right_attr["roles"]) != 1 or left_attr["roles"] == right_attr["roles"]:
        raise RuntimeError(f"ambiguous GOLD cut {parent['parent_id']}:{position}")
    words = load_ami_words(meeting)
    left_role = left_attr["roles"][0]
    right_role = right_attr["roles"][0]
    left_candidates = [word for word in words if word["role"] == left_role and word["start_src"] < left_token["source_end_sample"] and word["end_src"] > left_token["source_start_sample"]]
    right_candidates = [word for word in words if word["role"] == right_role and word["start_src"] < right_token["source_end_sample"] and word["end_src"] > right_token["source_start_sample"]]
    if not left_candidates or not right_candidates:
        raise RuntimeError(f"missing GT witness {parent['parent_id']}:{position}")
    left_word = max(left_candidates, key=lambda row: (row["end_src"], row["start_src"]))
    right_word = min(right_candidates, key=lambda row: (row["start_src"], row["end_src"]))
    if left_word["end_src"] > right_word["start_src"]:
        raise RuntimeError(f"overlapping GOLD witness {parent['parent_id']}:{position}")
    return {
        "token_position": position,
        "char_offset": sum(len(token["text"]) for token in parent["tokens"][:position]),
        "left": {key: left_word[key] for key in ("id", "role", "text", "start_src", "end_src")},
        "right": {key: right_word[key] for key in ("id", "role", "text", "start_src", "end_src")},
        "gt_gap_samples": right_word["start_src"] - left_word["end_src"],
        "agent_checked_unambiguous_sequential_word_boundary": True,
    }


def validate_response(response: Any, expected_ids: list[str]) -> tuple[bool, str | None]:
    if not isinstance(response, dict) or response.get("schema") != "meaning-first-e1-output-1" or set(response) != {"schema", "translations"}:
        return False, "invalid_top_level_schema"
    rows = response.get("translations")
    if not isinstance(rows, list):
        return False, "translations_not_array"
    ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "text"} or not isinstance(row.get("id"), str) or not isinstance(row.get("text"), str) or not row["text"].strip():
            return False, "invalid_translation_item"
        ids.append(row["id"])
    if len(ids) != len(set(ids)):
        return False, "duplicate_id"
    if ids != expected_ids:
        if set(ids) == set(expected_ids) and len(ids) == len(expected_ids):
            return False, "reordered_id"
        return False, "missing_or_extra_id"
    return True, None


def prepare() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    current: dict[str, tuple[str, dict[str, Any]]] = {}
    all_requests: list[tuple[str, int, str, dict[str, Any]]] = []
    header = None
    for row in iter_bundle(POLICY_BUNDLE):
        if row.get("type") == "header":
            header = row
        if row.get("type") != "parent" or row.get("cohort") != "current":
            continue
        parent = row["value"]
        current[parent["parent_id"]] = (row["meeting"], parent)
        for request in parent.get("requests", []):
            all_requests.append((row["meeting"], parent["index"], parent["parent_id"], request))
    translations = {row["parent_id"]: row for row in iter_bundle(TRANSLATION_BUNDLE) if row.get("type") == "parent"}
    split_ids = sorted(parent_id for parent_id, (_, parent) in current.items() if parent.get("text") and len(parent["r2"]["child_ids"]) > 1)
    if len(current) != 2482 or len(split_ids) != 20 or set(split_ids) != set(ASSESSMENTS):
        raise RuntimeError("current parent or split-parent census mismatch")
    records: list[dict[str, Any]] = []
    ratings = Counter()
    for parent_id in split_ids:
        meeting, parent = current[parent_id]
        translated = translations[parent_id]
        rating, evidence, mismatch, distinction = ASSESSMENTS[parent_id]
        ratings[rating] += 1
        records.append({
            "record_type": "E0_split_parent",
            "meeting": meeting,
            "cluster_id": parent["cluster_id"],
            "parent_index": parent["index"],
            "parent_id": parent_id,
            "source_span_samples": parent["span"],
            "accepted_text": parent["text"],
            "gt_supported_role_or_status_runs": role_runs(parent),
            "r0": {
                "texts": parent["r0"]["child_texts"],
                "retained_outcomes": translated["retained_r0"],
                "contaminated_attributable_characters": parent["r0"]["contamination"]["contaminated_chars"],
            },
            "r2": {
                "texts": [child["text"] for child in translated["r2_children"]],
                "retained_outcomes": [translation_outcome(child) for child in translated["r2_children"]],
                "contaminated_attributable_characters": parent["r2"]["contamination"]["contaminated_chars"],
            },
            "contamination_delta_r2_minus_r0_characters": parent["r2"]["contamination"]["contaminated_chars"] - parent["r0"]["contamination"]["contaminated_chars"],
            "exploratory_unblinded_korean_assessment": {
                "main_meaning": rating,
                "main_evidence_certainty": "low_fragment_diagnostic_not_semantic_proof" if parent_id in E0_LOW_CERTAINTY_FRAGMENT_IDS else "case_specific_unblinded_agent_judgment",
                "quoted_decisive_evidence": evidence,
                "main_meaning_basis": mismatch,
                "short_response_vs_broken_fragment": {
                    "classification": "contains_legitimate_short_responses_and_broken_fragments" if parent_id in SHORT_RESPONSE_MIXED_IDS else distinction,
                    "rule": "A complete short speech act aligned to a role change is not harm merely because it is short. A cut inside a grammatical or semantic unit is a broken fragment.",
                    "observed": "This parent contains legitimate short responses; broken grouping is reported separately and is not itself counted as semantic harm." if parent_id in SHORT_RESPONSE_MIXED_IDS else "Broken grouping is a readability/fragment diagnostic and is not itself sufficient to establish main-meaning harm.",
                },
                "readability_fragment_diagnostic": "worse readability from isolated function words or fragments; main meaning rated independently",
                "causal_interpretation": "Boundary changed and each child was translated independently with empty context; these records do not isolate boundary representation from independent-generation variance.",
                "independence": "Human/agent exploratory read of existing outputs; unblinded and not an independent rating.",
            },
        })
    controls = defaultdict(list)
    for meeting, index, parent_id, request in all_requests:
        if request.get("text") not in CONTROL_TEXTS:
            continue
        body = historical_body(request)
        fingerprint = digest_bytes(canonical(body).encode("utf-8"))
        translated = translations[parent_id]
        child = next(item for item in translated["r2_children"] if item["original_child_id"] == request["utterance_id"])
        controls[(request["text"], fingerprint)].append({"meeting": meeting, "parent_index": index, "parent_id": parent_id, "child_id": request["utterance_id"], "original_request_id": request["id"], "outcome": translation_outcome(child)})
    if {key[0] for key, value in controls.items() if len(value) >= 2} != set(CONTROL_TEXTS):
        raise RuntimeError("generation-control census mismatch")
    for (text, fingerprint), occurrences in sorted(controls.items()):
        if len(occurrences) < 2:
            continue
        records.append({
            "record_type": "E0_reconstructed_identical_request_control",
            "source_text": text,
            "canonical_reconstructed_request_sha256": fingerprint,
            "occurrences": occurrences,
            "observed_distinct_responses": sorted({item["outcome"].get("response") for item in occurrences if item["outcome"].get("response") is not None}),
            "identity_limit": "The retained request fields plus pinned builder reconstruct equal complete request bodies, but historical request_body_sha256 values and deleted acquisition journals are unavailable; actual wire-byte equality is therefore uncertain and this is not labeled an exact-wire control.",
        })
    cases = []
    physical_requests: dict[str, dict[str, Any]] = {}
    for case_id, category, parent_id, cuts, facts in E1_SPECS:
        meeting, parent = current[parent_id]
        witnesses = [boundary_witness(parent, meeting, cut) for cut in cuts]
        boundaries = [0, *cuts, len(parent["tokens"])]
        gold_texts = ["".join(token["text"] for token in parent["tokens"][start:end]) for start, end in zip(boundaries, boundaries[1:])]
        if "".join(gold_texts) != parent["text"] or "".join(token["text"] for token in parent["tokens"]) != parent["text"]:
            raise RuntimeError(f"text conservation failure {parent_id}")
        b_segments = [{"id": "P0", "text": parent["text"]}]
        g_segments = [{"id": f"P{index}", "text": text} for index, text in enumerate(gold_texts)]
        arm_rows = {}
        for arm, segments in (("B", b_segments), ("G", g_segments)):
            logical = {"schema": "meaning-first-e1-input-1", "parent_text": parent["text"], "segments": segments}
            body = provider_body(logical)
            fingerprint = digest_bytes(canonical(body).encode("utf-8"))
            physical_requests.setdefault(fingerprint, body)
            arm_rows[arm] = {"logical_payload": logical, "provider_request_body": body, "canonical_request_sha256": fingerprint, "expected_output_ids": [segment["id"] for segment in segments]}
        cases.append({
            "record_type": "E1_frozen_case",
            "case_id": case_id,
            "category": category,
            "meeting": meeting,
            "cluster_id": parent["cluster_id"],
            "parent_index": parent["index"],
            "parent_id": parent_id,
            "source_span_samples": parent["span"],
            "accepted_text": parent["text"],
            "accepted_text_sha256": digest_bytes(parent["text"].encode("utf-8")),
            "token_provenance": [{"token_id": token["token_id"], "text": token["text"], "timing": token["timing"], "source_start_sample": token.get("source_start_sample"), "source_end_sample": token.get("source_end_sample")} for token in parent["tokens"]],
            "source_gt_facts_frozen_before_generation": facts,
            "gold_boundaries": witnesses,
            "gold_limit": "Offline noncausal human/agent-checked positional utility. No GT role/name/fact is included in either model payload. No GT repair changes accepted ASR text.",
            "annotation_identities": annotation_identity(meeting),
            "arms": arm_rows,
        })
    records.extend(cases)
    logical_calls = len(cases) * 2
    plan = {
        "schema": "MEANING-FIRST-PROBE-PLAN-1",
        "baseline": BASELINE,
        "changed_agreement": "E0 calibrated exploratory read of all 20 current splits, then E1 exactly two B/G arms over 12 source-first frozen DEV cases revised before any new output; E2 only conditionally after a positive E1 and separate authorization.",
        "authorization": {"local_only": True, "paid": False, "api_llm_calls": False, "native_asr_model_runs": False, "holdout": False, "installs": False},
        "retained_inputs": {"policy": {"path": str(POLICY_BUNDLE.relative_to(ROOT)), "sha256": digest_file(POLICY_BUNDLE)}, "translations": {"path": str(TRANSLATION_BUNDLE.relative_to(ROOT)), "sha256": digest_file(TRANSLATION_BUNDLE)}, "header": header, "historical_request_source_anchors": [{"path": "experiments/psem_r2_policy/live_runner.py", "facts": "BudgetedOpenRouter request settings and historical max_tokens=100"}, {"path": "experiments/psem_r2_policy/arms.py", "facts": "R2 prompt rendering, empty context, and null scene participant count"}, {"path": "experiments/psem_r2_policy/audio-runtime-af26d1d3.tar.gz:src/puripuly_heart/providers/llm/openrouter.py", "facts": "request body, routing, response validation, finish_reason length failure"}, {"path": "experiments/psem_r2_policy/audio-runtime-af26d1d3.tar.gz:src/puripuly_heart/providers/llm/messages.py", "facts": "user message wrapper"}, {"path": "experiments/psem_r2_policy/audio-runtime-af26d1d3.tar.gz:src/puripuly_heart/core/orchestrator/translation_turn.py", "facts": "child construction and source conservation"}, {"path": "experiments/psem_r2_policy/audio-runtime-af26d1d3.tar.gz:prompts/translation_prompt.md", "sha256": "7fac71f6539878d07d03e52ce560723d09c07f24d773cd2b0930c3d8f747726a"}]},
        "E0": {"scope": "all actually split current-R2 nonempty DEV parents", "split_parent_count": len(split_ids), "assessment_counts": dict(sorted(ratings.items())), "assessment_revision": {"superseded_candidate": "d6ab23c3336add2357147be5debc85e9ccd177a2", "prior_counts": {"worse": 18, "equal": 1, "unjudgeable": 1}, "reason": "Separate main-meaning error from fragment/readability cost; three fragment-only cases move to equal, d928e75b moves to worse for question-to-statement modality, and 9a75496 arithmetic wording is neutralized while retaining the observed loss of 35."}, "controls": list(CONTROL_TEXTS), "interpretation_limits": ["Purity/contamination is not semantic quality.", "Fragmentation/readability cost is not itself proof of main-meaning harm.", "Changed boundaries, loss of whole-parent context through independent child calls, and stochastic generation are not isolated by these historical outcomes.", "Ratings are unblinded exploratory human/agent readings, not independent blinded ratings."]},
        "E1": {
            "case_count": len(cases),
            "category_counts": dict(sorted(Counter(case["category"] for case in cases).items())),
            "selection_rule": "Use only current DEV accepted text, retained token/source timing, and original AMI GT before inspecting future output. Include the fixed semantic patterns and guards; require every GOLD cut to lie between mapped adjacent accepted tokens whose single-role GT attributions differ and whose directly overlapping GT witness words are sequential/nonoverlapping. Preserve unresolved timing and overlap guards unsplit. For the revised A-B-C quota, include the sole checked ES2002 candidate for cross-group coverage, then retain the earlier-source-interval ES2009 candidate. IDs are fixed in E1_SPECS order, never ranked by historical or future Korean output.",
            "selection_revision": {"supersedes_candidate": "d6ab23c3336add2357147be5debc85e9ccd177a2", "timing": "before generation; no new E1 output exists", "change": "Replace ES2009d C2 parent 8b32ec2a-298e-4326-a111-81e597eaa8e0 with ES2002b C2 parent 4bd08541-357d-4a3c-9a9a-637080b7a2aa to improve cross-group A-B-C coverage while preserving 12 cases and category counts.", "candidate_audit": {"scope_limit": "Checked retained current parents reported with exactly three eligible role segments; not a claim that every broader semantic opportunity in the corpus was exhaustively enumerated.", "A_B_A": {"eligible_checked": ["9a75496a-26b7-42f8-a41b-f6ed81d41dcc", "0294cb49-e5c7-4daa-9a44-839e6fde9b88", "9c8a8144-7606-4c05-b8f7-bed1cd38abbc"], "selected": ["9a75496a-26b7-42f8-a41b-f6ed81d41dcc", "0294cb49-e5c7-4daa-9a44-839e6fde9b88"], "set_aside": {"9c8a8144-7606-4c05-b8f7-bed1cd38abbc": "two-case quota filled by the earlier meeting/source-order candidates"}}, "A_B_C": {"eligible_checked": ["63954a9b-e069-41bc-821f-f82a248b0855", "8b32ec2a-298e-4326-a111-81e597eaa8e0", "4bd08541-357d-4a3c-9a9a-637080b7a2aa"], "selected": ["63954a9b-e069-41bc-821f-f82a248b0855", "4bd08541-357d-4a3c-9a9a-637080b7a2aa"], "set_aside": {"8b32ec2a-298e-4326-a111-81e597eaa8e0": "after cross-group inclusion, later ES2009d source interval than selected 63954a9b"}}}, "historical_read_overlap": "R1 parent 9a75496a is also among E0's unblinded historical reads. The freeze precedes NEW outputs and is source-first, not historically naive; the overlap is disclosed rather than hidden by a swap."},
            "selection_gap": "In the checked exact-three-role candidate scan, A-B-A has three eligible parents, all ES2009; A-B-C has three, two ES2009d and one ES2002b. The revised targets span ES2009 and ES2002 in both the question/answer and A-B-C categories. EN2009 contributes a same-speaker guard but no target. Meeting-wide strata are not treated as parent-level semantic proof, and the candidate audit does not claim broader exhaustive corpus availability.",
            "arms": {"B": "whole parent as one positional segment", "G": "same parent text partitioned only at frozen GOLD positional boundaries"},
            "shared_prompt": PROMPT,
            "prompt_provenance": "The E1 meaning-preserving parent-batch system prompt is newly authored for this probe. Only the request envelope and confirmed model/settings are reused. Historical VRChat-prompt-conditioned R0/R2 Korean outputs are descriptive E0 evidence and are not pairwise-comparable E1 arm evidence.",
            "provider_confirmation": {"source": "pinned runtime archive af26d1d3 and retained current requests", "model": MODEL, "temperature": 0.6, "reasoning": {"effort": "none"}, "provider": PROVIDER["provider"], "source_language": "en", "target_language": "ko", "context": "", "scene_participant_count": None, "historical_max_tokens": 100, "proposed_E1_max_tokens_both_arms": MAX_TOKENS, "ceiling_change_requires_approval": True, "ceiling_rationale": "The selected maximum source is 124 characters and G can require three nonempty Korean strings plus JSON/IDs. Historical 100 tokens is a credible structured-output truncation risk. The proposed symmetric 512-token bound follows a separate product_translation probe precedent only as a sizing precedent, not reused quality evidence; it is one shared bound for B/G, not an adaptive retry or model/provider change."},
            "request_scope": {"logical_call_upper_bound": logical_calls, "unique_canonical_requests_after_dedup": len(physical_requests), "dedup_rule": "One future output per identical canonical request fingerprint; bind it to both logical arms when B and G payloads are identical.", "automatic_retries": 0, "decision_sensitive_repeats": "not authorized; require separate bounded approval", "maximum_selected_source_characters": max(len(case["accepted_text"]) for case in cases)},
            "output_contract": {"schema": "meaning-first-e1-output-1", "structural_validator_responsibility": "validate_response checks exact schema, nonempty string values, and exact requested IDs in order only.", "structural_failures": ["invalid schema", "empty output item", "missing ID", "duplicate ID", "extra ID", "reordered ID"], "future_generation_runner_responsibility_not_implemented_here": "Require terminal provider success; non-200, missing choices, empty content, finish_reason=length, truncation, and invalid JSON are failures and are never converted to source_only success. No live failure flow is implemented or claimed by this local preparer.", "future_rater_responsibility": "Judge meaning, adequacy, invention, omission, source echo, and proper-name preservation against source/facts. Do not use a contains-Hangul or text-not-equal-source heuristic.", "difference_from_issue_160": "This standalone probe deliberately rejects reordered IDs. Issue #160 R5 permits product result-ID reordering normalization; this probe does not redefine that product contract."},
            "evaluation": {"partial_blinding": "Future rater receives source, fact table, shared context, arm outputs, and visible segment counts; names/expectations and arm identity are hidden. Segment count prevents full blinding.", "arm_mapping": "Create only at generation time with fresh private randomness, never a deterministic scientific seed; keep mapping separate from rater.", "main": ["improve", "equal", "worse", "unjudgeable"], "readability": "separate", "supplementary": {"adequacy": "0-3", "faithfulness": "0-3", "korean_fluency": "0-3"}, "severe": ["actual negation/agreement reversal", "wrong claim ownership", "omission", "invention"], "guard_maintenance_absolute_check": "S1/S2/U1/O1 are no-cut controls whose B/G requests deduplicate to one shared output. They pass only when that output is structurally valid, preserves every frozen fact including negation/agreement and overlap ambiguity, and has no severe omission, invention, negation reversal, or false ownership. A shared tie is not an automatic pass.", "investigation_only": "First-reference new-wrong-token flag is not automatically actual product harm.", "progress_threshold": ">=3 clear meaning improvements spanning >=2 independent groups, zero new severe errors, and all four absolute guard checks pass; equal/style-only are not wins; report declines and gaps explicitly."},
        },
        "E2_conditional_not_implemented": {
            "gate": "E1 positive plus separate authorization",
            "sources": "Two independent short ~3 minute DEV sources; default ES2009d first 180s plus a suitable other group.",
            "source_state": "Source-zero continuous state, or explicit new-session warmup with no claim of restored state.",
            "fixed_stack": {"asr": "existing Deepgram unchanged", "native": {"chunk": 6, "left": 1, "right": 7, "fifo": 188, "cache": 188, "update": 144, "threshold": 0.5, "format": "F16", "executable_and_model": "existing pins", "confirmation": "existing"}, "audio": "#159 current product 7s hard seal; historical 6s remains separate; no Audio change"},
            "arms": {"U13-low": "snapshot comparison baseline", "D": "raw native-slot/UNKNOWN label runs without CURRENT/OTHER compression; not a Soniox reproduction", "L": "LOCAL-CUT+BATCH-1: locally confirmed singleton speaker-change boundary only, same generation, received by t0, unique lexical gap compatible within ±80ms, adjacent local support; reject only ambiguous/straddle/overlap boundary; UNKNOWN/coverage change alone does not cut and unrelated timing unknown is not global veto; preserve actual A-B-A/A-B-C boundaries"},
            "causality": "Freeze t0 before PSEM waiting and receiver sequence; computation cannot consume later messages.",
            "logging_if_authorized": ["source blocks/feed", "ASR send/accept", "parent seal", "t0", "snapshot", "source frontier", "native emit/receipt/confirmation", "boundary reason", "dispatch/first visible/completion", "entry/eviction"],
            "translation_scope": "Only selected <=12 new parents, B/G/distinct candidate inputs, deduplicated; partial load, not full product latency.",
            "witness_tracking": [{"parent_id": "c06fe9cd-e419-47bd-8427-066896b59e18", "cohort": "old", "meeting": "ES2009d", "parent_index": 340, "source_span_samples": [27872768, 27959728], "approx_source_seconds": [1742.048, 1747.483], "planned_180s_reachability": "outside; historical regression context only, do not expand the short capture merely to reach it and do not claim a new-run verification", "acoustic_transition": "preceding lexical role D to tail token 24 role B", "reference_change": "tail token 24 text ‘ Me’ remains GT role B, but its R2 UNKNOWN-4 unit reference is D whereas the whole R0 unit reference is B", "interpretation_limit": "Earlier standalone fragment ‘My’ is distinct from the actual guard token. This is a scorer reference-change flag only, with no model/UI relation claim and no retrospective safety approval."}, {"parent_id": "8465ae6c-5f27-457f-bcc1-42944ff0294e", "cohort": "current", "approx_source_seconds": 107, "known_reference_flag": None}, {"parent_id": "684dd105-8c15-498b-9848-4911f3a5f825", "cohort": "current", "approx_source_seconds": 78, "known_reference_flag": None}],
            "timing": "Primary zero added wait. If useful but late, log max(0, required_receipt-t0); no 400ms adoption from old interpolation.",
            "decision": ["D sufficient => choose simpler D", "D overfragments and L preserves meaning => L", "right boundary not meaning-helpful => representation not model", "late => time-budget decision", "model bottleneck only if named"],
            "prohibited": ["automatic H training", "sweeps", "HOLDOUT", "model/provider changes", "#160 changes", "fake early evidence or assumed new witness IDs/text"],
        },
        "reproduction": {"prepare_and_verify": "python experiments/psem_r2_policy/artifacts/meaning-first-probe/prepare.py all", "verify_only": "python experiments/psem_r2_policy/artifacts/meaning-first-probe/prepare.py verify"},
    }
    decision = {
        "schema": "MEANING-FIRST-PROBE-DECISION-1",
        "baseline": BASELINE,
        "status": "local_preparation_complete_generation_blocked",
        "E0_decision": "The calibrated unblinded read contains no meaning improvement: 16 cases are rated worse in exploratory case judgments, three fragment-dominant cases are equal on main meaning with worse readability, and one is unjudgeable. The 16 are not claimed as independently proven harms or a fixed count of clear semantic failures; contamination reductions do not imply semantic improvement. This limited read does not select a policy, replace U13, satisfy historical gates, unlock HOLDOUT, or complete #156/#160.",
        "E0_counts": {"split_parents": len(split_ids), "assessment": dict(sorted(ratings.items())), "generation_controls": len(CONTROL_TEXTS)},
        "E1_freeze": {"cases": len(cases), "logical_call_upper_bound": logical_calls, "unique_canonical_requests_after_dedup": len(physical_requests), "historical_max_tokens": 100, "proposed_shared_max_tokens": MAX_TOKENS, "results_created": False, "supersedes_candidate": "d6ab23c3336add2357147be5debc85e9ccd177a2", "replacement": "C2 8b32ec2a -> 4bd08541 before generation"},
        "historical_criteria": "Unfulfilled and unchanged: independent-group count, confirmatory interval/improvement/safety, timing readiness, semantic ratings, HOLDOUT, and end-to-end product evidence.",
        "one_next_action": "Request explicit approval for frozen E1 generation, including the declared shared max_tokens increase from 100 to 512; until then make no API/LLM calls.",
        "dependency_gaps": ["Paid/new generation approval is absent, including approval of the shared 512-token output ceiling.", "Fresh private arm mapping must be created only if generation is approved.", "Only two independent groups contribute target opportunities; EN2009 appears only as a guard.", "Historical wire-byte request hashes are unavailable, so E0 reconstructed-equality controls remain explicitly uncertain."],
    }
    return plan, records, decision


def write_outputs(plan: dict[str, Any], records: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "cases.jsonl").write_text("".join(canonical(record) + "\n" for record in records), encoding="utf-8")
    (HERE / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify() -> dict[str, Any]:
    plan, expected_records, decision = prepare()
    actual_plan = json.loads((HERE / "plan.json").read_text(encoding="utf-8"))
    actual_records = [json.loads(line) for line in (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line]
    actual_decision = json.loads((HERE / "decision.json").read_text(encoding="utf-8"))
    if actual_plan != plan or actual_records != expected_records or actual_decision != decision:
        raise RuntimeError("durable output differs from deterministic preparation")
    e0 = [row for row in actual_records if row["record_type"] == "E0_split_parent"]
    e1 = [row for row in actual_records if row["record_type"] == "E1_frozen_case"]
    if len(e0) != 20 or len(e1) != 12 or len({row["parent_id"] for row in e1}) != 12:
        raise RuntimeError("coverage mismatch")
    expected_case_ids = ["Q1", "Q2", "Q3", "Q4", "R1", "R2", "C1", "C2", "S1", "S2", "U1", "O1"]
    if [row["case_id"] for row in e1] != expected_case_ids:
        raise RuntimeError("frozen case identity/order mismatch")
    c2 = next(row for row in e1 if row["case_id"] == "C2")
    if c2["parent_id"] != "4bd08541-357d-4a3c-9a9a-637080b7a2aa" or [boundary["left"]["role"] for boundary in c2["gold_boundaries"]] != ["C", "D"] or [boundary["right"]["role"] for boundary in c2["gold_boundaries"]] != ["D", "B"]:
        raise RuntimeError("revised C2 identity or GOLD role sequence mismatch")
    if any(row["parent_id"] == "8b32ec2a-298e-4326-a111-81e597eaa8e0" for row in e1):
        raise RuntimeError("superseded C2 remains frozen")
    if Counter(row["exploratory_unblinded_korean_assessment"]["main_meaning"] for row in e0) != Counter({"worse": 16, "equal": 3, "unjudgeable": 1}):
        raise RuntimeError("calibrated E0 assessment mismatch")
    s2 = next(row for row in e1 if row["case_id"] == "S2")
    if "as fixing this problem by means of the mono microphone socket" not in s2["source_gt_facts_frozen_before_generation"]["facts"][0]:
        raise RuntimeError("S2 fact drift")
    for case in e1:
        text = case["accepted_text"]
        if "".join(segment["text"] for segment in case["arms"]["G"]["logical_payload"]["segments"]) != text:
            raise RuntimeError("G concatenation mismatch")
        if "".join(segment["text"] for segment in case["arms"]["B"]["logical_payload"]["segments"]) != text:
            raise RuntimeError("B concatenation mismatch")
        for arm in ("B", "G"):
            request = case["arms"][arm]
            if digest_bytes(canonical(request["provider_request_body"]).encode("utf-8")) != request["canonical_request_sha256"]:
                raise RuntimeError("request fingerprint mismatch")
    guards = [case for case in e1 if case["category"].startswith("guard_")]
    if len(guards) != 4 or any(case["arms"]["B"]["canonical_request_sha256"] != case["arms"]["G"]["canonical_request_sha256"] for case in guards):
        raise RuntimeError("no-cut guard dedup mismatch")
    if "A shared tie is not an automatic pass." not in plan["E1"]["evaluation"]["guard_maintenance_absolute_check"]:
        raise RuntimeError("absolute guard contract missing")
    contract = plan["E1"]["output_contract"]
    if "not_implemented_here" not in " ".join(contract) or "contains-Hangul" not in contract["future_rater_responsibility"] or "newly authored" not in plan["E1"]["prompt_provenance"]:
        raise RuntimeError("output or prompt responsibility contract mismatch")
    witness = plan["E2_conditional_not_implemented"]["witness_tracking"][0]
    if witness["cohort"] != "old" or witness["acoustic_transition"] != "preceding lexical role D to tail token 24 role B" or not witness["planned_180s_reachability"].startswith("outside"):
        raise RuntimeError("historical witness calibration mismatch")
    sample = e1[0]["arms"]["G"]["expected_output_ids"]
    response = {"schema": "meaning-first-e1-output-1", "translations": [{"id": item, "text": "번역"} for item in sample]}
    if validate_response(response, sample) != (True, None):
        raise RuntimeError("valid response rejected")
    failure_checks = {
        "missing": validate_response({**response, "translations": response["translations"][:-1]}, sample)[1],
        "duplicate": validate_response({**response, "translations": [response["translations"][0], response["translations"][0], *response["translations"][2:]]}, sample)[1],
        "reorder": validate_response({**response, "translations": list(reversed(response["translations"]))}, sample)[1],
        "extra": validate_response({**response, "translations": [*response["translations"], {"id": "EXTRA", "text": "번역"}]}, sample)[1],
        "empty": validate_response({**response, "translations": [{**response["translations"][0], "text": " "}, *response["translations"][1:]]}, sample)[1],
        "invalid_schema": validate_response({"schema": "wrong", "translations": response["translations"]}, sample)[1],
    }
    if failure_checks != {"missing": "missing_or_extra_id", "duplicate": "duplicate_id", "reorder": "reordered_id", "extra": "missing_or_extra_id", "empty": "invalid_translation_item", "invalid_schema": "invalid_top_level_schema"}:
        raise RuntimeError(f"failure contract mismatch: {failure_checks}")
    sizes = {name: (HERE / name).stat().st_size for name in ("plan.json", "cases.jsonl", "decision.json", "prepare.py")}
    if sum(sizes.values()) > 1_000_000:
        raise RuntimeError("artifact set is not compact")
    return {"ok": True, "e0_split_parents": len(e0), "e0_assessment_counts": dict(sorted(Counter(row["exploratory_unblinded_korean_assessment"]["main_meaning"] for row in e0).items())), "e1_cases": len(e1), "case_ids": [row["case_id"] for row in e1], "parent_ids": [row["parent_id"] for row in e1], "category_counts": plan["E1"]["category_counts"], "logical_call_upper_bound": plan["E1"]["request_scope"]["logical_call_upper_bound"], "unique_canonical_requests_after_dedup": plan["E1"]["request_scope"]["unique_canonical_requests_after_dedup"], "revised_C2": {"parent_id": c2["parent_id"], "gold_role_sequence": ["C", "D", "B"]}, "guard_shared_request_count": len(guards), "failure_contract_checks": failure_checks, "sizes": sizes, "total_bytes": sum(sizes.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "all"), default="all", nargs="?")
    args = parser.parse_args()
    if args.command in {"prepare", "all"}:
        write_outputs(*prepare())
    if args.command in {"verify", "all"}:
        print(canonical(verify()))


if __name__ == "__main__":
    main()
