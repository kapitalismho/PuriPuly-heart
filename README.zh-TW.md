<p align="center">
  <img src="src/puripuly_heart/data/icons/icon.png" alt="PuriPuly <3" width="128" />
</p>

<h1 align="center">PuriPuly <3</h1>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.2.2-blue" alt="Version" />
  <img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue" alt="License: AGPL-3.0-or-later" />
  <img src="https://img.shields.io/badge/python-3.12-yellow" alt="Python" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey" alt="Platform" />
</p>

<p align="center">適用於 VRChat 的 LLM 雙向翻譯器</p>

<h2 align="center">
  <a href="README.md">🇺🇸 English</a> ·
  <a href="README.ar.md">🇸🇦 العربية</a> ·
  <a href="README.bg.md">🇧🇬 Български</a> ·
  <a href="README.ca.md">CA Català</a> ·
  <a href="README.cs.md">🇨🇿 Čeština</a> ·
  <a href="README.da.md">🇩🇰 Dansk</a> ·
  <a href="README.de.md">🇩🇪 Deutsch</a> ·
  <a href="README.el.md">🇬🇷 Ελληνικά</a> ·
  <a href="README.es.md">🇪🇸 Español</a> ·
  <a href="README.et.md">🇪🇪 Eesti</a> ·
  <a href="README.fi.md">🇫🇮 Suomi</a> ·
  <a href="README.fr.md">🇫🇷 Français</a> ·
  <a href="README.hi.md">🇮🇳 हिन्दी</a> ·
  <a href="README.hu.md">🇭🇺 Magyar</a> ·
  <a href="README.id.md">🇮🇩 Bahasa Indonesia</a> ·
  <a href="README.it.md">🇮🇹 Italiano</a> ·
  <a href="README.ja.md">🇯🇵 日本語</a> ·
  <a href="README.ko.md">🇰🇷 한국어</a> ·
  <a href="README.lt.md">🇱🇹 Lietuvių</a> ·
  <a href="README.lv.md">🇱🇻 Latviešu</a> ·
  <a href="README.ms.md">🇲🇾 Bahasa Melayu</a> ·
  <a href="README.nl.md">🇳🇱 Nederlands</a> ·
  <a href="README.no.md">🇳🇴 Norsk</a> ·
  <a href="README.pl.md">🇵🇱 Polski</a> ·
  <a href="README.pt.md">🇵🇹 Português</a> ·
  <a href="README.ro.md">🇷🇴 Română</a> ·
  <a href="README.ru.md">🇷🇺 Русский</a> ·
  <a href="README.sk.md">🇸🇰 Slovenčina</a> ·
  <a href="README.sv.md">🇸🇪 Svenska</a> ·
  <a href="README.th.md">🇹🇭 ไทย</a> ·
  <a href="README.tr.md">🇹🇷 Türkçe</a> ·
  <a href="README.uk.md">🇺🇦 Українська</a> ·
  <a href="README.vi.md">🇻🇳 Tiếng Việt</a> ·
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> ·
  🇹🇼 繁體中文
</h2>

> ⚠️ **這是一個可攜式分支** [kapitalismho/PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart)。為便於分發和修改而進行調整。[下載可攜版 ←](../../releases)

---

## 示範

![PuriPuly 與 VRCT 翻譯結果比較。](docs/images/demo/ko-en_screenshot.png)

---

<video src="https://github.com/user-attachments/assets/c667f44d-b91d-42a9-b24a-e6a993b392d3" controls width="100%"></video>

透過 PuriPuly 的更多真實交流範例：
- [示範 1](https://www.youtube.com/watch?v=3p0CamYui0o)
- [示範 2](https://youtu.be/DoX36Y7J_lc?si=YjbeVTS8v3jGQB1w)
- [示範 3](https://www.youtube.com/watch?v=D0npvp68xNY)

---

## 終於，像真正的朋友一樣交談。

你曾經歷過那個場景。想安慰朋友，卻只能說：「你還好嗎？」

你知道「翻譯器」無法傳達你心中的話。所以我做了一個可以的。

- **LLM 驅動的翻譯** — 行話、口語、正式和非正式用語 — 全部自然呈現。
- **上下文記憶** — 對話在了解先前上下文的情況下自然流暢。
- **雙向語音翻譯** — 也翻譯對方的語音，支援 VR 字幕。
- **透過 Discord 啟動** — 無需複雜設定即可開始。

## 常見問題

- **翻譯品質如何？**
→ 使用 Gemma 4，結果比 DeepL 好 6 倍。

- **需要多長時間？**
→ 使用 Gemma 4 和雲端 STT，延遲約一秒半。

- **需要付費嗎？**
→ 是的，但不是立即。新用戶獲得免費額度。之後價格非常低 — 每美元數千句。

- **需要 API 金鑰嗎？**
→ 是的，但不是立即。只需安裝並透過 Discord 驗證。

- **對方語音翻譯如何運作？**
→ 在安靜環境中一對一對話效果最佳。在 VRChat 中使用 Earmuff。

- **語音辨識不佳/緩慢。**
→ 使用雲端 STT。在 Intel 上僅使用效能核心。

- **資料如何處理？**
→ 語音和對話內容本機儲存，不會傳送到伺服器。

### [📥 下載](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)

---

## 翻譯比較
![透過 Gemba MQM 的翻譯品質圖表。](docs/images/performance/1.png)

- 使用 Microsoft Gemba MQM 進行實驗。
- 多輪環境以實現更真實的對話。
- 完整結果[在此](https://github.com/kapitalismho/korean-llm-context-translation-benchmark)。

## 費用

### 每美元使用次數

| LLM \ ASR | Qwen ASR (本地) | Qwen ASR (雲端) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | 14,380 | 2,920 | 3,710 | 1,180 |
| **DeepSeek V4 Flash** | 19,410 | 3,080 | 3,980 | 1,210 |

### 免費額度

| 服務 | 免費額度 | 期限 | 備註 |
|--------|------------|------|------|
| **Deepgram** | $200 | 無限制 | - |
| **Google AI Studio** | $10 | 1 年 | Gemini 訂閱者每月 |
| **Alibaba Cloud** | 每模型 1M tokens | 90 天 | 新加坡區域 |
| **Cerebras** | 每日 1M tokens | 無限制 | 限制 5 次呼叫/分鐘 |

---

## 使用方式

1. 從[下載頁面](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)下載最新版本。
2. 安裝 PuriPuly。
3. 按 **STT**。
4. 按 **TRANS** 並透過 Discord 驗證。
5. 按 **Subtitles** 啟用 VR 字幕。
6. 按 **Peer** 翻譯對方語音。
7. 在 VRChat 中啟用 OSC：選單 ← 設定 ← OSC ← 啟用。

---

## 開發

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e '.[dev]'
python -m puripuly_heart.main run-gui
```

```bash
black src tests
ruff check src tests
python -m pytest
```

---

## 開發者
[salee](https://github.com/kapitalismho)

## 授權條款
[AGPL-3.0-or-later](LICENSE)
