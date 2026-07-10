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

---

### 如果音訊擷取無法運作
如果音訊擷取無法運作，請開啟**設定 > 一般**並按照以下步驟操作。

1. 將 **Audio Host API** 變更為 **Auto** 或 **MME**。
2. 選擇正確的麥克風。
3. 重新啟動應用程式。

---

### 中國使用者注意事項

如果您所在地區的 Soniox/Gemini/Deepgram 被封鎖，請使用以下組合：

- STT: **Qwen ASR**
- LLM: **DeepSeek V4 Flash**

   > 您可以透過 QQ 而非 Discord 進行驗證。

---

### 使用自己的 API 金鑰

按照您想使用的服務的指南操作。

對於翻譯，我們推薦透過 OpenRouter 使用 Gemma 4 模型。

同時設定 ASR — PuriPuly 搭配雲端 STT 能提供最佳體驗。
即使使用相同的 Qwen ASR，本地和雲端辨識的差異也很明顯。

我們建議從 Deepgram 開始。
光是註冊就能獲得 $200 免費額度。

<details>
<summary><h3>OpenRouter</h3></summary>

1. Set the options inside the red circle as shown in the screenshot.
   ![step0](docs/images/openrouter/0.png)

2. In the app, click the button inside the red circle.
   ![step1](docs/images/openrouter/1.png)

3. Login at OpenRouter.
   ![step2](docs/images/openrouter/2.png)

4. Click the button inside the red circle to exit the payment screen.
   ![step3](docs/images/openrouter/3.png)

5. Click the **Authorize** button.
   ![step4](docs/images/openrouter/4.png)

6. Prepay as much as you plan to use.
   ![step5](docs/images/openrouter/5.png)

<details>
<summary><h3>如果 Authorize 按鈕沒有作用</h3></summary>

如果您點擊了 Authorize 但仍無法驗證，請重試，或直接建立 API 金鑰：

6. Click your account in the top right, go to the API Keys tab on the left, then click the Create button in the center.
   ![step6](docs/images/openrouter/6.png)

7. Click the Create button.
   ![step7](docs/images/openrouter/7.png)

8. Click the button to copy the API key, then paste it into the API tab of the translator.
   ![step8](docs/images/openrouter/8.png)

</details>

</details>

<details>
<summary><h3>DeepSeek</h3></summary>

1. Set the options inside the red circle as shown in the screenshot.
   ![step0](docs/images/deepseek/0.png)

2. Go to the [DeepSeek official homepage](https://www.deepseek.com/en/) and click the **Access API** button.
   ![step1](docs/images/deepseek/1.png)

3. Login on the homepage.
   ![step2](docs/images/deepseek/2.png)

4. Go to the API Keys tab and click **Create new API Keys**.
   ![step3](docs/images/deepseek/3.png)

5. Click the button to copy the API key, then paste it into the API tab of the translator.
   ![step4](docs/images/deepseek/4.png)

6. Go to the Top Up tab and prepay as much as you plan to use.
   ![step5](docs/images/deepseek/5.png)

</details>

<details>
<summary><h3>Deepgram</h3></summary>

1. Login to the [Deepgram Console](https://console.deepgram.com/).
   ![step1](docs/images/deepgram/1.png)

2. If you see a welcome message/survey, click **Skip**.
   ![step2](docs/images/deepgram/2.png)

3. Select **STT (Speech-to-Text)** on the service selection screen.
   ![step3](docs/images/deepgram/3.png)

4. In the API Keys menu, click **Create a New API Key**.
   ![step4](docs/images/deepgram/4.png)

5. Enter a key name (e.g., `puripuly`) and create.
   ![step5](docs/images/deepgram/5.png)

6. Copy the generated key and paste it into PuriPuly settings.
   ![step6](docs/images/deepgram/6.png)

</details>

<details>
<summary><h3>Gemini</h3></summary>

1. Go to [Google AI Studio](https://aistudio.google.com/apikey) and click the **Get API key** button.
   ![step1](docs/images/gemini/1.png)

2. Create a new project.
   ![step2](docs/images/gemini/2.png)

3. Choose any name for the project.
   ![step3](docs/images/gemini/3.png)

4. Select the project you created and click **Create key**.
   ![step4](docs/images/gemini/4.png)

5. Click the circled area.
   ![step5](docs/images/gemini/5.png)

6. Click the circled area to copy the key.
   ![step6](docs/images/gemini/6.png)

7. (Recommended) Click the yellow **Set Up Billing** button to upgrade to the paid tier.
The tier transition may take a moment.
   ![step7](docs/images/gemini/7.png)

<details>
<summary><h3>Gemini 付費訂閱者適用</h3></summary>

8. Go to [Google Developer Program](https://developers.google.com/program/my-benefits) and join the program.
   ![step8](docs/images/gemini/8.png)

9. Select the paid tier project you set up in step 7.
   ![step9](docs/images/gemini/9.png)

</details>

</details>

<details>
<summary><h3>Qwen</h3></summary>

1. Access Alibaba Cloud Model Studio via the appropriate path for your region:
   - [Mainland China](https://bailian.console.aliyun.com/cn-beijing)
   - [Outside Mainland China](https://bailian.console.alibabacloud.com)

2. Login at the URL above. Make sure to select the correct Region for your API key (e.g., Beijing).
   ![step2](docs/images/qwen/1.png)

3. Click the **gear icon** in the top right.
   ![step3](docs/images/qwen/2.png)

4. Create a workspace and go to the **API-KEY** page.
   ![step4](docs/images/qwen/3.png)

5. Click **Create API Key**.
   ![step5](docs/images/qwen/4.png)

6. Assign an account and workspace, then click OK.
   ![step6](docs/images/qwen/5.png)

7. Click the circled area to copy the key.
   ![step7](docs/images/qwen/6.png)

</details>

<details>
<summary><h3>Soniox</h3></summary>

1. Login to [Soniox Console](https://console.soniox.com/).
   ![step1](docs/images/soniox/1.png)

2. Enter an organization name of your choice.
   ![step2](docs/images/soniox/2.png)

3. Click **Add Funds** to link a payment method.
   ![step3](docs/images/soniox/3.png)

4. Soniox requires prepaid credits. Once added, go to the **API Keys** menu.
   ![step4](docs/images/soniox/4.png)

5. Create a new API Key.
   ![step5](docs/images/soniox/5.png)

6. Copy the generated key and paste it into PuriPuly settings.
   ![step6](docs/images/soniox/6.png)

</details>

<details>
<summary><h3>Cerebras</h3></summary>

1. Go to [Cerebras](https://www.cerebras.ai/) and click **Get started**.
   ![step1](docs/images/cerebras/1.png)

2. Log in.
   ![step2](docs/images/cerebras/2.png)

3. Choose the plan you want. We recommend starting with the free tier.
   ![step3](docs/images/cerebras/3.png)

4. Copy the API key and paste it into PuriPuly.
   ![step4](docs/images/cerebras/4.png)

<details>
<summary><h3>切換到付費方案</h3></summary>

5. Go to the **Billing** tab.
   ![step5](docs/images/cerebras/5.png)

6. Enter your name.
   ![step6](docs/images/cerebras/6.png)

7. Add as much credit as you need.
   ![step7](docs/images/cerebras/7.png)

</details>

</details>

---

## 開發

### 環境摘要

| 領域 | 建議環境 |
|---|---|
| Python 應用程式 | Windows |
| VR 覆蓋層 | Windows |
| Broker 服務 | Linux / WSL |

### Python 應用程式

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

```bash
# pip
pip install -e '.[dev]'

# or uv
uv sync --dev
```

```bash
pre-commit install
```

### 執行 GUI

```bash
# After activating venv
python -m puripuly_heart.main run-gui

# or run via uv
uv run python -m puripuly_heart.main run-gui
```

```bash
# 顯示隱藏的 UI 供檢查
python -m puripuly_heart.main run-gui --debug-ui-preview
```

### 測試與 Linting

```bash
black src tests          # 格式化
ruff check src tests     # Linting
python -m pytest         # 測試（建議在 venv 中）
```

### VR 覆蓋層

VR 字幕覆蓋層是從 `native/overlay/` 中的 Rust 專案建構的。

```powershell
cargo test --manifest-path native/overlay/Cargo.toml -q

cargo build `
  --manifest-path native/overlay/Cargo.toml `
  --locked `
  --release `
  --bin PuriPulyHeartOverlay `
  --target-dir target

New-Item -ItemType Directory -Force -Path build/overlay | Out-Null
Copy-Item target/release/PuriPulyHeartOverlay.exe build/overlay/PuriPulyHeartOverlay.exe -Force
Copy-Item third_party/openvr/win64/openvr_api.dll build/overlay/openvr_api.dll -Force

.\build\overlay\PuriPulyHeartOverlay.exe --check-startup-contract
```

### Broker 服務

詳見 `broker/README.md`。

```bash
pnpm install --frozen-lockfile
pnpm run typecheck
pnpm exec vitest run
pnpm --filter @puripuly-heart/broker run verify:config
pnpm --filter @puripuly-heart/broker run dev
```

---

## 開發者

[salee](https://github.com/kapitalismho)

---

## 貢獻者

[RICHARDwuxiaofei](https://github.com/RICHARDwuxiaofei)

---

## 特別感謝

SUI\_32C, Nagikokoro, motoka96, \_Ykol魚, kascr\_, Just Monika V, FLUVIA, Han โชเล่ย์, EA\_PE, Ephedrine, ~ eri ~

---

## 授權條款

[AGPL-3.0-or-later](LICENSE)

第三方授權條款和聲明： `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`
