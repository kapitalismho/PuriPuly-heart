<p align="center">
  <img src="src/puripuly_heart/data/icons/icon.png" alt="PuriPuly <3" width="128" />
</p>

<h1 align="center">PuriPuly <3</h1>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.4.0-blue" alt="Version" />
  <img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue" alt="License: AGPL-3.0-or-later" />
  <img src="https://img.shields.io/badge/python-3.12-yellow" alt="Python" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey" alt="Platform" />
</p>

<p align="center">LLM-based two-way translator for VRChat</p>

<h2 align="center">
  🇺🇸 English ·
  <a href="README.ko.md">🇰🇷 한국어</a> ·
  <a href="README.ja.md">🇯🇵 日本語</a> ·
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README.ru.md">🇷🇺 Русский</a>
</h2>

---

## Demo

![Comparison of translation results between PuriPuly (Deepgram + Gemini 3 Flash) and VRCT (Google Web Speech + Google Translate). PuriPuly STT: "아역시혼자기대하면안된다니깐", Translation: "(See, I knew I shouldn't have gotten my hopes up.)" | VRCT STT: "아 역시 혼자 기대하면 안 된다니까", Translation: "Oh, I guess you shouldn't expect it alone."](docs/images/demo/ko-en_screenshot.png)

---

<video src="https://github.com/user-attachments/assets/c667f44d-b91d-42a9-b24a-e6a993b392d3" controls width="100%"></video>

If you want to see more of actual communication with other foreign friends through PuriPuly:
- [Demo 1](https://www.youtube.com/watch?v=3p0CamYui0o)
- [Demo 2](https://youtu.be/DoX36Y7J_lc?si=YjbeVTS8v3jGQB1w)
- [Demo 3](https://www.youtube.com/watch?v=D0npvp68xNY)

---

## Finally, talk like real friends.

You've been there.  
Wanting to comfort a friend,  
but only managing: "Are you okay?"

You already know a 'translator'  
can't carry what's truly in your heart.

So I built one that can.

- **LLM-Powered Localization** — Slang, colloquialisms, and casual/formal speech, all rendered naturally.
- **Context Memory** — Keeps the conversation flowing naturally with awareness of prior context.
- **Two-way Voice Translation** — Translates the other person's voice too, with VR subtitle overlay support.
- **Start via Discord** — Get going right away without a complex setup process.

## Q&A

- **How good is the translation quality?**
→ When both you and the other person use this translator, you can have even the deepest kinds of conversations. Quantitatively, with Gemma 4 it scored 6× better than DeepL. See the 'Translation Comparison' section below for details.

- **How long does it take from speaking to getting a translation?**
→ With Gemma 4 and a cloud STT service, latency is typically in the mid-to-late 1-second range.

- **Does it cost money to use?**
→ Yes, but only later. New users get a free usage allowance, and even after that the pricing is very cheap; you can use it thousands of times for $1.

- **Do I need to get an API key?**
→ Yes, but again, only later. At first, just install and authenticate via Discord to start using it.

- **How polished is the feature for translating the other person's voice?**
→ It works best for one-on-one conversations in low-noise environments. Up to three people may be okay, but usability is not guaranteed. When using it in VRChat, use Earmuff to control the environment.

- **Voice recognition is poor / slow.**
→ If you're using local Qwen ASR, we recommend switching to a cloud STT service. If you're on Intel, configure PuriPuly so it's pinned to P-cores only.

- **How are voice and conversation contents handled?**
→ Voice and conversation contents are stored locally and are not sent to Puripuly servers. Other people's voices, transcripts, and translation results are never recorded. That said, the STT service and translation provider may process the data.

### [📥 Download](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)

---

## Translation Comparison
![Translation quality benchmark chart. It shows the mean error penalty per sentence (lower is better) evaluated using the Gemba MQM framework (judge model: Gemini 3.1 Pro Preview) on 216 multi-turn Korean to EN, JA, and ZH-Hans samples. Scores: Gemini 3.1 Flash-lite 0.573, Gemini 3 Flash 0.596, Gemma 4 26B A4B 0.813, Qwen 3.5 Plus 0.958, DeepSeek V4 Flash 1.025, Gemma 4 26B A4B (no-context) 1.265, DeepSeek V4 Flash (no-context) 1.647, Qwen 3.5 Flash 2.198, DeepL 4.963, DeepL (no-context) 5.717, Google Translation Basic 5.998.](docs/images/performance/1.png)

- We ran the experiment using Microsoft's Gemba MQM framework.
- It was set up as a multi-turn environment to better resemble real conversation.
- For the full results, see [here](https://github.com/kapitalismho/korean-llm-context-translation-benchmark).

## Cost

### Uses per Dollar

#### Recommended Models

| LLM \ ASR | Qwen ASR (Local) | Qwen ASR (Cloud) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B + 31B** | 14,380 | 2,920 | 3,710 | 1,180 |
| **DeepSeek V4 Flash** | 19,410 | 3,080 | 3,980 | 1,210 |

#### Other Models

| LLM \ ASR | Qwen ASR (Local) | Qwen ASR (Cloud) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | 14,380 | 2,920 | 3,710 | 1,180 |
| **Gemma 4 31B (OpenRouter)** | 13,700 | 2,780 | 3,530 | 1,120 |
| **Gemma 4 31B (Cerebras)** | 920 | 730 | 770 | 540 |
| **Gemini 3 Flash** | 1,710 | 1,170 | 1,280 | 740 |
| **Gemini 3.1 Flash-Lite** | 3,430 | 1,770 | 2,030 | 940 |
| **Qwen 3.5 Plus** | 7,460 | 2,460 | — | — |
| **Local LLMs** | Unlimited | 3,660 | 5,000 | 1,290 |

### Cost per Utterance

#### Recommended Models

| LLM \ ASR | Qwen ASR (Local) | Qwen ASR (Cloud) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B + 31B** | ~$0.00007 | ~$0.0003 | ~$0.0003 | ~$0.0008 |
| **DeepSeek V4 Flash** | ~$0.00005 | ~$0.0003 | ~$0.0003 | ~$0.0008 |

#### Other Models

| LLM \ ASR | Qwen ASR (Local) | Qwen ASR (Cloud) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | ~$0.00007 | ~$0.0003 | ~$0.0003 | ~$0.0008 |
| **Gemma 4 31B (OpenRouter)** | ~$0.00007 | ~$0.0003 | ~$0.0003 | ~$0.0009 |
| **Gemma 4 31B (Cerebras)** | ~$0.0011 | ~$0.0014 | ~$0.0013 | ~$0.0019 |
| **Gemini 3 Flash** | ~$0.0006 | ~$0.0009 | ~$0.0008 | ~$0.0014 |
| **Gemini 3.1 Flash-Lite** | ~$0.0003 | ~$0.0006 | ~$0.0005 | ~$0.0011 |
| **Qwen 3.5 Plus** | ~$0.0001 | ~$0.0004 | — | — |
| **Local LLMs** | $0 | ~$0.0003 | ~$0.0002 | ~$0.0008 |

*   *Based on (Input 900 tokens + Output 12 tokens) × 1.2 avg LLM calls per utterance.*
*   *Uses per Dollar is derived from the un-rounded values in the Cost per Utterance table.*
*   *All costs and usage counts are approximate.*
*   *DeepSeek assumes a 70% cache hit rate.*
*   *Qwen API costs are based on the Beijing region.*
*   *Pricing as of May 25, 2026 / Fast Response mode active.*

### Free Credits

| Service | Free Credit | Duration | Note |
|--------|------------|------|------|
| **Deepgram** | $200 | None | - |
| **Google AI Studio** | $10 | 1 year | Monthly for Gemini subscribers |
| **Alibaba Cloud** | 1M tokens per model | 90 days | Singapore region |
| **Alibaba Cloud** | ¥300 | 1 year | Students in China |
| **Cerebras** | 1M tokens daily | None | 5 calls per minute limit |

---

# If you run into problems or anything feels unclear, feel free to DM me on [Twitter/X](https://x.com/kapitalismho).

## Usage

1. Download the latest version from the [Download page](https://github.com/kapitalismho/PuriPuly-heart/releases/latest).
2. Install PuriPuly.
3. Click the **TALK** button.
4. Click the **TRANS** button, then authenticate via Discord.
5. Click the **CAPTIONS** button to turn on VR subtitles.
6. (Optional) Click the **LISTEN** button to enable translation of the other person's voice.

   > Peer voice translation needs a low-noise space to work properly. When using it in VRChat, use Earmuff to control the environment.

7. Enable OSC in VRChat: Action menu → Settings → OSC → Enable.

For bidirectional control setup and the stable parameter ABI, see [VRChat OSC controls](docs/vrchat-osc.md).

### If audio capture does not work
If audio capture does not work, open **Settings > General** and follow these steps.

1. Change **Audio Host API** to **Auto** or **MME**.
2. Select the correct microphone.
3. Restart the app.

---

### Note for Users in China

If Soniox/Gemini/Deepgram are blocked in your region, please use the following combination:

- STT: **Qwen ASR**
- LLM: **DeepSeek V4 Flash**

   > You can authenticate through QQ instead of Discord.

---

### Using Your Own API Keys

Follow the guide that matches the service you want to use.

For the translation LLM, we recommend using the Gemma 4 model through OpenRouter.

By the way, while you're setting things up, why not configure ASR too?
PuriPuly delivers the best experience when paired with a cloud STT.
For instance, even with the same Qwen ASR, local and cloud voice-recognition performance differ noticeably.

We recommend starting with Deepgram.
Just signing up gets you $200 in free credits.

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
<summary><h3>If clicking Authorize didn't authenticate you</h3></summary>

If you clicked Authorize but you're still not authenticated, retry, or directly issue an API key as below and paste it in.

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
<summary><h3>For Gemini paid subscribers</h3></summary>

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
<summary><h3>To switch to the paid tier</h3></summary>

5. Go to the **Billing** tab.
   ![step5](docs/images/cerebras/5.png)

6. Enter your name.
   ![step6](docs/images/cerebras/6.png)

7. Add as much credit as you need.
   ![step7](docs/images/cerebras/7.png)

</details>

</details>

---

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Roadmap

Upcoming work is tracked publicly on the [PuriPuly project board](https://github.com/users/kapitalismho/projects/2).

---

## Development

### Environments

| Surface                    | Recommended environment | Documentation                                          |
| -------------------------- | ----------------------- | ------------------------------------------------------ |
| Python desktop application | Windows                 | This section                                           |
| Broker service             | Linux                   | [`broker/README.md`](broker/README.md)                 |
| Native VR overlay          | Windows                 | [`native/overlay/README.md`](native/overlay/README.md) |

### Python Environment

The Python application requires Python 3.12 or 3.13.

Create and activate the Windows environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the application and development dependencies:

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

`uv` may be used instead:

```powershell
uv sync --dev
```

Install the repository hooks:

```powershell
pre-commit install
```

For Linux or WSL work, use `.venv-wsl` when it is available.

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv sync --dev
```

Repositories configured with `direnv` may run commands through:

```bash
direnv exec . <command>
```

### Running the Application

Run the Flet desktop application:

```powershell
python -m puripuly_heart.main run-gui
```

The equivalent `uv` command is:

```powershell
uv run python -m puripuly_heart.main run-gui
```

Developer preview controls for hidden UI states are enabled with:

```powershell
python -m puripuly_heart.main run-gui --debug-ui-preview
```

### Python Verification

Format the Python sources and tests:

```powershell
black src tests
```

Check formatting without modifying files:

```powershell
black --check src tests
```

Run lint checks:

```powershell
ruff check src tests
```

Run the complete Python test suite:

```powershell
python -m pytest
```

Run a focused test file or directory during development:

```powershell
python -m pytest tests/path/to/test_file.py
```

### Other Surfaces

Broker documentation is maintained in [`broker/README.md`](broker/README.md).

Native VR overlay documentation is maintained in [`native/overlay/README.md`](native/overlay/README.md).

Custom HTTP API extension documentation is maintained in [`docs/http-extensions.md`](docs/http-extensions.md). For the JSON Schema required for connection, see [`docs/http-extension.schema.json`](docs/http-extension.schema.json).

VRChat OSC controls are documented in [`docs/vrchat-osc.md`](docs/vrchat-osc.md).

---

## Developer

[salee](https://github.com/kapitalismho)

---

## Contributors

[RICHARDwuxiaofei](https://github.com/RICHARDwuxiaofei)
[fzcfweasdferttgg-png](https://github.com/fzcfweasdferttgg-png)

---

## Special Thanks

SUI\_32C, Nagikokoro, motoka96, \_Ykol魚, kascr\_, Just Monika V, FLUVIA, Han โชเล่ย์, EA\_PE, Ephedrine, ~ eri ~, fzcfweasdferttgg-png, Welcius, nunu299

---

## License

[AGPL-3.0-or-later](LICENSE)

Third-party licenses and notices: `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`
