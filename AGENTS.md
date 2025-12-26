PuriPuly <3 Agent Instructions
===========================================

Project Overview
----------------

PuriPuly <3 is a VRChat real-time speech translation pipeline:
audio/VAD -> STT -> LLM -> OSC. It ships a GUI (Flet) and CLI entry points.

Development Environment
-----------------------

- Python >=3.12,<3.14 (see `pyproject.toml`).
- Install: `pip install -e .` (dev: `pip install -e '.[dev]'`).
- Entry point: `python -m ajebal_daera_translator.main ...`

Repository Structure
--------------------

- `src/ajebal_daera_translator/app/`: CLI and headless runners (stdin/mic).
- `src/ajebal_daera_translator/core/`: orchestrator, audio, VAD, OSC, STT, LLM.
- `src/ajebal_daera_translator/providers/`: concrete STT/LLM providers.
- `src/ajebal_daera_translator/ui/`: Flet UI, views, and controller.
- `src/ajebal_daera_translator/config/`: settings, prompts, and paths.
- `prompts/`: system prompt files.
- `tests/` and `tests/integration/`: unit and opt-in integration tests.

Common Tasks
------------

- Send one message: `python -m ajebal_daera_translator.main osc-send "hello"`.
- Stream stdin: `python -m ajebal_daera_translator.main run-stdin [--use-llm]`.
- Mic pipeline: `python -m ajebal_daera_translator.main run-mic [--use-llm]`.

Configuration and Settings
--------------------------

- Settings schema and validation: `src/ajebal_daera_translator/config/settings.py`.
- Default settings path: `src/ajebal_daera_translator/config/paths.py`.
- Settings are JSON; `to_dict`/`from_dict` must stay in sync.
- If you add or change settings:
  - Update dataclasses and validation in `settings.py`.
  - Update `to_dict`/`from_dict`.
  - Wire UI fields in `src/ajebal_daera_translator/ui/views/settings.py`.

Prompts
-------

- Prompt loader: `src/ajebal_daera_translator/config/prompts.py`.
- Prompts are loaded from `prompts/` or `AJEBAL_DAERA_PROMPTS_DIR`.
- Provider-specific prompts: `prompts/gemini.txt`, `prompts/qwen.txt`.
- If you add a new LLM provider or a default prompt, add a matching file.

Providers
---------

- STT interface: `src/ajebal_daera_translator/core/stt/backend.py`.
- LLM interface: `src/ajebal_daera_translator/core/llm/provider.py`.
- Implement providers under:
  - `src/ajebal_daera_translator/providers/stt/`
  - `src/ajebal_daera_translator/providers/llm/`
- Update enums and settings in `src/ajebal_daera_translator/config/settings.py`.
- Wire provider selection in `src/ajebal_daera_translator/app/wiring.py` if needed.
- Add unit tests in `tests/` and integration tests in `tests/integration/`.

Orchestrator / Hub
------------------

- Core pipeline coordinator is `ClientHub` in `src/ajebal_daera_translator/core/orchestrator/hub.py`.
- Flow: audio/VAD events -> STT -> (LLM translation) -> OSC queue -> UI events.
- Owns task lifecycles for STT event loop and OSC flushing; `start()`/`stop()` manage cancellation and provider shutdown.

Context Memory
--------------

- Implemented in `ClientHub` as `_translation_history`.
- Defaults: `context_time_window_s = 20.0`, `context_max_entries = 3`.
- Only recent entries within the time window are formatted and passed to the LLM.
- Update `tests/test_context_memory.py` when changing window size or behavior.

Async Patterns
--------------

- Keep I/O and provider calls async; avoid blocking the event loop.
- Run long-lived loops with `asyncio.create_task` and ensure they are cancelled on shutdown.
- Always `await` provider `close()` in teardown paths.
- In UI, use `page.run_task(...)` for async work instead of blocking callbacks.

Team Rules
----------

- Branch strategy: trunk-based on `main` (no long-lived release branches).
- Merge strategy: squash (one commit per PR/topic).
- Use short-lived topic branches when helpful; keep `main` releasable.
- Release by tagging `main` with `vX.Y.Z`.
- Style tools: format with `black`, lint with `ruff` via `pre-commit`.
- Release scope: bump version, tag, and produce PyInstaller builds.

Secrets and Security
--------------------

- SecretStore reads from keyring or encrypted file, then falls back to env vars.
- Known keys and env vars are documented in `README.md`.
- If `secrets.backend = "encrypted_file"`, require `AJEBAL_SECRETS_PASSPHRASE`.
- Never commit real credentials or API keys.

Testing
-------

- Unit tests: `pytest` (defaults in `pyproject.toml`).
- Integration tests are required when provider behavior changes (STT/LLM/OSC, provider wiring, or prompts).
- Integration tests are opt-in otherwise: `INTEGRATION=1 python -m pytest`.
- Integration tests require provider credentials; see `README.md`.

Build and Distribution
----------------------

- PyInstaller builds are configured in `build.spec`.
- Windows installer is configured in `installer.iss`.
- Package data lives under `src/ajebal_daera_translator/data/`.

Release Checklist
-----------------

- Bump versions in `pyproject.toml`, `src/ajebal_daera_translator/__init__.py`, and `installer.iss` (MyAppVersion).
- Run `pre-commit run --all-files` and `pytest` (plus integration tests if provider behavior changed).
- Build the app: `pyinstaller build.spec`.
- Build the installer: `ISCC installer.iss` (Windows/Inno Setup).
- Tag the release on `main`: `git tag vX.Y.Z` and publish artifacts.

Provider Change Checklist
-------------------------

- Implement provider under `src/ajebal_daera_translator/providers/` to keep the shared interface working across the app.
- Update enums/defaults/validation in `src/ajebal_daera_translator/config/settings.py` so settings can select and persist it.
- Wire UI choices in `src/ajebal_daera_translator/ui/views/settings.py` so users can configure it.
- Update provider wiring in `src/ajebal_daera_translator/app/wiring.py` so runtime selection resolves to the new class.
- Add LLM prompts in `prompts/{provider}.txt` to keep output quality consistent (LLM providers).
- Add PyInstaller hidden imports in `build.spec` so bundled builds include dynamic provider modules.
- Document secrets/env vars in `README.md` so users can run it without guesswork.
- Add dependencies in `pyproject.toml` to avoid runtime import errors.
- Run unit tests and required integration tests to catch external API regressions.

Config Backward Compatibility Rules
-----------------------------------

- Always supply defaults for new settings so existing `settings.json` continues to load.
- If a setting is renamed, accept the old key in `from_dict` and map it to the new one.
- Keep enum fallbacks in `from_dict` for removed/invalid values.
- Update `to_dict` and `from_dict` together to avoid mismatched schemas.
- For major schema changes, add a version key and document migration rules.
