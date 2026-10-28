## How to download
Download `{{INSTALLER_EXE}}` from the Assets section below, then run it.

## How to use
- [English](https://github.com/kapitalismho/PuriPuly-heart/blob/dev/README.md#using-your-own-api-keys)
- [한국어](https://github.com/kapitalismho/PuriPuly-heart/blob/dev/docs/readme/README.ko.md#자신의-api-키-사용하기)
- [日本語](https://github.com/kapitalismho/PuriPuly-heart/blob/dev/docs/readme/README.ja.md#自分のapiキーを使う)
- [简体中文](https://github.com/kapitalismho/PuriPuly-heart/blob/dev/docs/readme/README.zh-CN.md#使用您自己的-api-密钥)
- [Русский](https://github.com/kapitalismho/PuriPuly-heart/blob/dev/docs/readme/README.ru.md#свои-api-ключи)


## Change logs
- CPU speech recognition now uses Orukeet for English, Fun-ASR-Nano for Chinese, and Fun-ASR-MLT-Nano for Korean. Japanese Parakeet CTC remains available.
- Fixed CPU Auto's multilingual routing regression: all 25 Orukeet languages and all 31 Fun-ASR-MLT-Nano languages are considered. English, Chinese and Japanese keep their existing models; Orukeet wins other overlaps, covering 37 language codes in total.
- Talk and Listen use independent transcribe.cpp CPU Model/Session instances with two inference threads each. GPU ASR keeps one shared resident Model, separate channel Sessions and serialized inference, including safe model replacement.
- Old CPU provider selections migrate automatically. Recognized retired Qwen0.6/Parakeet v3 assets are cleaned after recoverable settings migration; blocked deletion retries on later starts without removing unknown or Japanese/GPU assets.
- The replacement CPU models support pinned, integrity-checked downloads from Hugging Face and the same-byte China mirror.
- The window stays responsive during startup, including the first Windows process scan. Closing the window during startup also drains pending preparation safely.
- Gemini speech recognition preserves eligible pre-roll when resuming a stream after an idle period.
- From the second launch, the message input shows a randomly selected prompt or usage tip. Changing the interface language keeps the selected tip.
- Local translation models now share one Local section in the model picker. The GPT-6 luna description also clarifies that it can use a ChatGPT plan.
- ASR logs identify applied provider and model selections, final latency remains visible when an already-sent preview becomes final, and routine Output success messages no longer flood the logs.

---
**If PuriPuly has made your world a bit wider,
please consider hitting the ⭐ Star button at the top of the GitHub page.
It would mean a lot**
