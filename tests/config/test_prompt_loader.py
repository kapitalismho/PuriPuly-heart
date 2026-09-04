from __future__ import annotations

import sys
from pathlib import Path

import pytest

import puripuly_heart.config.prompts as prompts_module
from puripuly_heart.config.prompts import (
    build_translation_prompt_variables,
    get_default_prompt,
    get_prompts_dir,
    get_translation_prompt_template,
    list_prompts,
    load_prompt,
    load_prompt_for_provider,
    render_translation_prompt_template,
    warm_prompt_cache,
)


def test_load_prompt_for_llm_providers_uses_shared_translation_prompt() -> None:
    raw = Path("prompts/translation_prompt.md").read_text(encoding="utf-8").strip()
    assert load_prompt_for_provider("gemini") == raw
    assert load_prompt_for_provider("qwen") == raw
    assert load_prompt_for_provider("deepseek") == raw
    assert load_prompt_for_provider("openrouter") == raw
    assert load_prompt_for_provider("local_llm") == raw


def test_local_llm_uses_shared_translation_prompt() -> None:
    assert load_prompt_for_provider("local_llm") == get_translation_prompt_template()


def test_render_translation_prompt_uses_exact_korean_to_english_rules_and_examples() -> None:
    template = get_translation_prompt_template()

    rendered = render_translation_prompt_template(
        template,
        source_name="Korean",
        target_name="English",
    )

    assert "Interpret the Korean text to translate into English" in rendered
    assert "Korean" in rendered
    assert "English" in rendered
    assert "Use contractions" in rendered
    assert "Context Use Example" in rendered
    assert "아까 그분 목소리" in rendered
    assert "### Target language Rules" in rendered
    assert "## Examples" in rendered
    assert rendered.index("### Target language Rules") < rendered.index("## Examples")
    assert rendered.index("## Examples") < rendered.index("## Output")
    assert "${" not in rendered


def test_unknown_source_to_english_uses_fallback_examples() -> None:
    template = get_translation_prompt_template()

    rendered = render_translation_prompt_template(
        template,
        source_name="French",
        target_name="English",
    )

    assert "J'ai trouvé un nouvel avatar." in rendered
    assert "I'm gonna go make some coffee." in rendered


def test_chinese_traditional_example_selection_does_not_reuse_simplified_examples() -> None:
    template = get_translation_prompt_template()

    english_rendered = render_translation_prompt_template(
        template,
        source_name="Chinese Traditional",
        target_name="English",
    )
    korean_variables = build_translation_prompt_variables(
        "Chinese Traditional",
        "Korean",
    )

    assert "J'ai trouvé un nouvel avatar." in english_rendered
    assert "刚才在那边弹吉他的那个人挺厉害的" not in english_rendered
    assert korean_variables["translationExamples"] == ""
    assert korean_variables["translationExamplesSection"] == ""


def test_chinese_target_variants_use_shared_chinese_rules() -> None:
    variables = build_translation_prompt_variables(
        "English",
        "Chinese Traditional (Taiwan)",
    )

    assert "Prefer natural softeners" in variables["targetLanguageRules"]


def test_warm_prompt_cache_prevents_request_time_file_reads(monkeypatch) -> None:
    warm_prompt_cache()

    def fail_read(_path: Path) -> str:
        raise AssertionError("prompt files should not be read after warm_prompt_cache()")

    monkeypatch.setattr(prompts_module, "_read_prompt_text", fail_read)

    rendered = render_translation_prompt_template(
        get_translation_prompt_template(),
        source_name="Japanese",
        target_name="Korean",
    )

    assert "Japanese" in rendered
    assert "Korean" in rendered
    assert "해요체" in rendered


def test_unspecified_source_uses_input_ref_and_skips_pair_examples() -> None:
    variables = build_translation_prompt_variables(
        "English",
        "Japanese",
        source_specified=False,
    )
    rendered = render_translation_prompt_template(
        get_translation_prompt_template(),
        source_name="English",
        target_name="Japanese",
        source_specified=False,
    )

    assert variables["sourceName"] == "<input>"
    assert variables["sourceTextRef"] == "<input>"
    assert variables["translationExamples"] == ""
    assert variables["translationExamplesSection"] == ""
    assert variables["targetLanguageRulesSection"].startswith("### Target language Rules\n")
    assert "Interpret <input> to translate into Japanese" in rendered
    assert "the English text" not in rendered
    assert "Context Use Example" not in rendered
    assert "## Examples" not in rendered
    assert "### Target language Rules" in rendered


def test_unspecified_source_to_english_skips_fallback_examples() -> None:
    variables = build_translation_prompt_variables(
        "French",
        "English",
        source_specified=False,
    )
    rendered = render_translation_prompt_template(
        get_translation_prompt_template(),
        source_name="French",
        target_name="English",
        source_specified=False,
    )

    assert variables["translationExamples"] == ""
    assert variables["translationExamplesSection"] == ""
    assert "Interpret <input> to translate into English" in rendered
    assert "J'ai trouvé un nouvel avatar." not in rendered
    assert "the French text" not in rendered
    assert "## Examples" not in rendered


def test_missing_rules_and_examples_omit_optional_section_headings() -> None:
    rendered = render_translation_prompt_template(
        get_translation_prompt_template(),
        source_name="English",
        target_name="French",
    )

    assert "### Target language Rules" not in rendered
    assert "## Examples" not in rendered
    assert "## Output" in rendered


def test_build_translation_prompt_variables_returns_expected_keys_and_content() -> None:
    variables = build_translation_prompt_variables("English", "Japanese")

    assert set(variables) == {
        "sourceName",
        "sourceTextRef",
        "targetName",
        "inputChannel",
        "targetLanguageRules",
        "targetLanguageRulesSection",
        "translationExamples",
        "translationExamplesSection",
    }
    assert variables["sourceName"] == "English"
    assert variables["sourceTextRef"] == "the English text"
    assert variables["targetName"] == "Japanese"
    assert variables["inputChannel"] == "self"
    assert "タメ口" in variables["targetLanguageRules"]
    assert variables["targetLanguageRulesSection"].startswith("### Target language Rules\n")
    assert "Context Use Example" in variables["translationExamples"]
    assert variables["translationExamplesSection"].startswith("## Examples\n")


def test_get_prompts_dir_prefers_env(tmp_path, monkeypatch) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.txt").write_text("DEFAULT", encoding="utf-8")

    monkeypatch.setenv("PURIPULY_HEART_PROMPTS_DIR", str(prompts_dir))

    assert get_prompts_dir() == prompts_dir


def test_list_prompts_returns_sorted_names(tmp_path, monkeypatch) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "b.md").write_text("B", encoding="utf-8")
    (prompts_dir / "a.md").write_text("A", encoding="utf-8")

    monkeypatch.setenv("PURIPULY_HEART_PROMPTS_DIR", str(prompts_dir))

    assert list_prompts() == ["a", "b"]


def test_load_prompt_falls_back_to_default(tmp_path, monkeypatch) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.txt").write_text("DEFAULT", encoding="utf-8")

    monkeypatch.setenv("PURIPULY_HEART_PROMPTS_DIR", str(prompts_dir))

    assert load_prompt("missing") == "DEFAULT"


def test_load_prompt_returns_empty_when_missing(tmp_path, monkeypatch) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    monkeypatch.setenv("PURIPULY_HEART_PROMPTS_DIR", str(prompts_dir))

    assert load_prompt("missing") == ""


def test_load_prompt_for_provider_requires_translation_prompt_for_llm_provider(
    tmp_path, monkeypatch
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.txt").write_text("DEFAULT", encoding="utf-8")

    monkeypatch.setenv("PURIPULY_HEART_PROMPTS_DIR", str(prompts_dir))

    with pytest.raises(FileNotFoundError):
        load_prompt_for_provider("gemini")


def test_get_prompts_dir_uses_pyinstaller_meipass(tmp_path, monkeypatch) -> None:
    bundle_root = tmp_path / "bundle"
    prompts_dir = bundle_root / "prompts"
    prompts_dir.mkdir(parents=True)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)

    assert get_prompts_dir() == prompts_dir


def test_list_prompts_returns_empty_when_missing(monkeypatch, tmp_path) -> None:
    missing_dir = tmp_path / "missing"
    monkeypatch.setattr(prompts_module, "get_prompts_dir", lambda: missing_dir)

    assert list_prompts() == []


def test_get_default_prompt_reads_translation_prompt(tmp_path, monkeypatch) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "translation_prompt.md").write_text("TRANSLATION", encoding="utf-8")

    monkeypatch.setenv("PURIPULY_HEART_PROMPTS_DIR", str(prompts_dir))

    assert get_default_prompt() == "TRANSLATION"


def test_get_prompts_dir_falls_back_to_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PURIPULY_HEART_PROMPTS_DIR", raising=False)
    monkeypatch.setattr(prompts_module, "__file__", str(tmp_path / "fake.py"))

    assert get_prompts_dir() == tmp_path / "prompts"
