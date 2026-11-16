from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import bench_photo_438 as bench


def _message(phase, text):
    message = {"type": "message", "content": [{"type": "output_text", "text": text}]}
    if phase is not None:
        message["phase"] = phase
    return message


def _events(messages):
    return [{"type": "response.completed", "response": {"status": "completed", "output": messages}}]


def _photo(tmp_path):
    return bench.PreparedPhoto("cHJvb2Y=", tmp_path / "source.png", 1280, 720, 1.0, 0, 0, 2.0)


def _record():
    return {"usage_raw": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "timing_ms": {"request_start_to_first_block": 123.0}, "parse_errors": []}


def _capture(directory, messages):
    bench.save(directory / "terminal-events.json", _events(messages))
    raw = "".join(part["text"] for message in messages for part in message["content"])
    (directory / "response.txt").write_text(raw, encoding="utf-8")


def test_final_blocks_exclude_valid_commentary_and_preserve_physical_lines(tmp_path):
    commentary = json.dumps({"blocks": [{"bbox": [1, 2, 3, 4], "source": "wrong", "translation": "wrong"}]})
    final = json.dumps({"blocks": [{"bbox": [10, 20, 100, 60], "source": "one\n\nthree", "translation": "고정"}]})
    _capture(tmp_path, [_message("commentary", commentary), _message("final_answer", final)])
    record = _record()

    bench.extract_response(tmp_path, _photo(tmp_path), "jpeg85", None, record)

    blocks = bench.load(tmp_path / "blocks.json")["blocks"]
    assert [(block["source"], block["source_line_count"]) for block in blocks] == [("one\n\nthree", 3)]
    assert record["structured_parse_status"] == "completed"
    assert record["timing_ms"]["request_start_to_first_block"] is None
    assert record["usage_raw"]["total_tokens"] == 120
    assert (tmp_path / "response.txt").read_text(encoding="utf-8") == commentary + final


def test_sole_final_preserves_observed_first_block_timing(tmp_path):
    final = '{"blocks":[{"bbox":[10,20,100,60],"source":"one","translation":"고정"}]}'
    _capture(tmp_path, [_message("final_answer", final)])
    record = _record()

    bench.extract_response(tmp_path, _photo(tmp_path), "png", None, record)

    assert record["structured_parse_status"] == "completed"
    assert record["timing_ms"]["request_start_to_first_block"] == 123.0


def test_quad_uses_explicit_final_not_malformed_commentary(tmp_path):
    final = '{"quads":[{"block_id":0,"quad":[[10,20],[100,22],[98,60],[11,58]]}]}'
    _capture(tmp_path, [_message("commentary", "{broken}} token-like text"), _message("final_answer", final)])
    record = _record()

    bench.extract_response(tmp_path, _photo(tmp_path), "quad", [{"block_id": 0}], record)

    assert record["geometry_available"] is True
    assert record["structured_parse_status"] == "completed"
    assert bench.load(tmp_path / "quad-original.json") == {"0": [[10, 20], [100, 22], [98, 60], [11, 58]]}
    assert record["timing_ms"]["request_start_to_first_block"] is None
    assert record["usage_raw"]["total_tokens"] == 120


@pytest.mark.parametrize("messages", [
    [],
    [_message(None, '{"blocks":[]}')],
    [_message("commentary", '{"blocks":[]}')],
    [_message("final_answer", '{"blocks":[]}'), _message("final_answer", "invalid")],
])
@pytest.mark.parametrize("arm", ["jpeg85", "quad"])
def test_missing_or_ambiguous_final_is_explicit_error_without_fallback(tmp_path, messages, arm):
    _capture(tmp_path, messages)
    bench.save(tmp_path / "blocks.json", {"blocks": ["stale"]})
    bench.save(tmp_path / "quad-original.json", {"0": "stale"})
    record = _record()

    bench.extract_response(tmp_path, _photo(tmp_path), arm, [{"block_id": 0}], record)

    assert record["structured_parse_status"] == "failed"
    assert "exactly one explicit final_answer" in record["phase_selection_error"]
    assert record["timing_ms"]["request_start_to_first_block"] is None
    assert not (tmp_path / "blocks.json").exists()
    assert record["usage_raw"]["total_tokens"] == 120
    assert not (tmp_path / "quad-original.json").exists()
    if arm == "quad":
        assert record["geometry_available"] is False
        assert record["unavailable_block_ids"] == ["0"]
        assert bench.load(tmp_path / "quads.json")["sidecar"] is None


def test_malformed_final_does_not_select_valid_commentary(tmp_path):
    _capture(tmp_path, [_message("commentary", '{"blocks":[]}'), _message("final_answer", "not JSON")])
    record = _record()

    bench.extract_response(tmp_path, _photo(tmp_path), "jpeg95", None, record)

    assert record["structured_parse_status"] == "failed"
    assert record["parse_errors"]
    assert not (tmp_path / "blocks.json").exists()


def test_missing_completed_terminal_is_rejected():
    with pytest.raises(ValueError, match="completed terminal response"):
        bench.select_final_answer([])


def test_reparse_never_accesses_secret_or_transport_and_retains_superseded_evidence(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reparse must not access secrets or transport")

    monkeypatch.setattr(bench.KeyringSecretStore, "get", forbidden)
    monkeypatch.setattr(bench.network_clients, "external_async_client", forbidden)
    bench.save(tmp_path / "corpus.json", {"images": [{"id": "image-01"}]})
    image = tmp_path / "image-01"
    source = tmp_path / "encoded.jpg"
    source.write_bytes(b"proof")
    meta = {"source_path": str(source), "source_dimensions": [1280, 720], "send_dimensions": [1280, 720],
            "scale": 1.0, "offset_x": 0, "offset_y": 0, "encodings": {
                arm: {"path": str(source), "preparation_ms": 2.0} for arm in bench.ARMS}}
    bench.save(image / "preparation/preparation.json", meta)
    for arm in (*bench.ARMS, "quad"):
        directory = image / "requests" / arm / "attempt-01"
        text = '{"quads":[]}' if arm == "quad" else '{"blocks":[]}'
        _capture(directory, [_message("final_answer", text)])
        record = {"image_id": "image-01", "arm": arm, "status": "completed", "encoded_bytes": 5,
                  "request_json_bytes": 10, **_record()}
        bench.save(directory / "measurement.json", record)
    args = SimpleNamespace(root=tmp_path, phase="reparse", image=None)

    asyncio.run(bench.run(args))

    result = bench.load(tmp_path / "phase-reextraction.json")
    assert len(result["rows"]) == 4
    assert all(row["structured_parse_status"] == "completed" for row in result["rows"])
    assert all(row["usage_unchanged"] for row in result["rows"])
    assert result["new_provider_calls"] == 0
    assert bench.load(image / "requests/jpeg85/attempt-01/superseded-concatenated-adapter/measurement.json")["timing_ms"]["request_start_to_first_block"] == 123.0
