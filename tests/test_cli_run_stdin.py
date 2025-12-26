from __future__ import annotations

from dataclasses import dataclass

import puripuly_heart.main as cli


@dataclass
class FakeRunner:
    settings: object
    llm: object | None
    last_llm: object | None = None

    async def run(self) -> int:
        FakeRunner.last_llm = self.llm
        return 0


def test_run_stdin_use_llm_wires_llm(monkeypatch, tmp_path):
    llm_obj = object()
    monkeypatch.setattr(cli, "HeadlessStdinRunner", FakeRunner)
    monkeypatch.setattr(cli, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(cli, "create_llm_provider", lambda *_a, **_k: llm_obj)

    code = cli.main(["--config", str(tmp_path / "settings.json"), "run-stdin", "--use-llm"])
    assert code == 0
    assert FakeRunner.last_llm is llm_obj


def test_run_stdin_use_llm_returns_error_on_init_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "HeadlessStdinRunner", FakeRunner)
    monkeypatch.setattr(cli, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr("builtins.print", lambda *_a, **_k: None)

    def _boom(*_a, **_k):
        raise ValueError("missing secret")

    monkeypatch.setattr(cli, "create_llm_provider", _boom)

    code = cli.main(["--config", str(tmp_path / "settings.json"), "run-stdin", "--use-llm"])
    assert code == 2
