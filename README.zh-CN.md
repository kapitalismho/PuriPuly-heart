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

<p align="center">LLM-based two-way translator for VRChat</p>

<h2 align="center">
  <a href="README.md">🇺🇸 English</a> ·
  <a href="README.ko.md">🇰🇷 한국어</a> ·
  <a href="README.ja.md">🇯🇵 日本語</a> ·
  🇨🇳 简体中文 ·
  <a href="README.ru.md">🇷🇺 Русский</a>
</h2>

---

## Demo

![PuriPuly（Deepgram + Qwen 3.5 Plus）与 VRCT（Google Web Speech + Google Translate）的翻译对比。PuriPuly 语音识别：“집에굴러다니는게손잡이빠진것까지합치면열개는넘을걸.”，翻译：“（家里散落的伞，加上那把没把手的，估计都超过十把了呢。）” | VRCT 语音识别：“집에 굴러다니는게 손잡이 빠진 것까지 합치면 10개는 넘을 걸”，翻译：“房子周围肯定有超过 10 个，包括那些没有把手的。”](docs/images/demo/ko-cn_screenshot.png)

---

<video src="https://github.com/user-attachments/assets/c667f44d-b91d-42a9-b24a-e6a993b392d3" controls width="100%"></video>

如果你想看更多通过 PuriPuly 与其他外国朋友实际交流的画面：
- [演示 1](https://www.youtube.com/watch?v=3p0CamYui0o)
- [演示 2](https://youtu.be/DoX36Y7J_lc?si=YjbeVTS8v3jGQB1w)
- [演示 3](https://www.youtube.com/watch?v=D0npvp68xNY)

---

## Finally, talk like real friends.

想安慰对方，
却只说得出一句"你还好吗？"
——对吧。

想传达的心意，
靠"翻译器"是传不到的，你知道的。

所以，我做了这个。

- **基于大语言模型的本地化** — 俚语、口语，乃至敬语与平语，都能自然转换
- **记忆上下文** — 结合语境，保持自然流畅的对话节奏
- **双向语音翻译** — 同时翻译对方的语音，支持 VR 字幕浮层
- **从 Discord 开始** — 无需复杂配置，立即可用

## 常见问题

- **翻译质量如何？**
→ 当对方与您都使用本翻译器时，足以支持最深入的对话。定量来说，以 Gemma 4 为例，比 DeepL 好 6 倍。详情请参阅下方"翻译比较"一节。

- **从说话到翻译完成需要多长时间？**
→ 以 Gemma 4 加云端 STT 服务为基准，延迟通常在 1 秒中后段。

- **使用需要付费吗？**
→ 是的，但要稍后才付。新用户会获得免费额度。即使额度用完，价格依然非常便宜——1 美元可使用数千次。

- **必须申请 API 密钥吗？**
→ 是的，但同样是稍后再说。一开始只需安装并通过 Discord 验证即可使用。

- **翻译对方语音的功能完成度如何？**
→ 在噪音较少的 1 对 1 环境下效果最好。三人对话也可能可用，但无法保证体验。若在 VRChat 中使用，请通过 Earmuff 功能控制环境。

- **语音识别不准 / 速度慢**
→ 如果您正在使用本地 Qwen ASR，建议改用云端 STT 服务。如果您是 Intel 用户，请将 PuriPuly 设置为仅固定分配到 P-core。

- **语音和对话内容如何处理？**
→ 语音和对话内容会保存在本地，不会发送到 Puripuly 服务器。此外，不会记录他人的语音、转写或翻译结果。但 STT 服务和翻译提供商可能会处理数据。

### [📥 下载](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)

---

## 翻译比较
![翻译质量基准测试图表。展示了使用 Gemba MQM 框架（评审模型：Gemini 3.1 Pro Preview）对 216 个韩语到英语、日语、中文简体多轮对话样本进行评估的单句平均错误惩罚分（越低越好）。各模型得分：Gemini 3.1 Flash-lite 0.573、Gemini 3 Flash 0.596、Gemma 4 26B A4B 0.813、Qwen 3.5 Plus 0.958、DeepSeek V4 Flash 1.025、Gemma 4 26B A4B (no-context) 1.265、DeepSeek V4 Flash (no-context) 1.647、Qwen 3.5 Flash 2.198、DeepL 4.963、DeepL (no-context) 5.717、Google Translation Basic 5.998。](docs/images/performance/1.png)

- 我们使用微软的 Gemba MQM 框架进行实验。
- 为贴近真实对话环境，采用了多轮对话设置。
- 完整实验结果请参阅[此处](https://github.com/kapitalismho/korean-llm-context-translation-benchmark)。

## 费用

### 每 1 美元可用次数

#### 推荐模型

| LLM \ ASR | Qwen ASR (本地) | Qwen ASR (云端) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | 14,380 次 | 2,920 次 | 3,710 次 | 1,180 次 |
| **DeepSeek V4 Flash** | 19,410 次 | 3,080 次 | 3,980 次 | 1,210 次 |

#### 其他模型

| LLM \ ASR | Qwen ASR (本地) | Qwen ASR (云端) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 31B (Cerebras)** | 920 次 | 730 次 | 770 次 | 540 次 |
| **DeepSeek V4 Pro** | 6,400 次 | 2,330 次 | 2,810 次 | 1,070 次 |
| **Gemini 3 Flash** | 1,710 次 | 1,170 次 | 1,280 次 | 740 次 |
| **Gemini 3.1 Flash-Lite** | 3,430 次 | 1,770 次 | 2,030 次 | 940 次 |
| **Qwen 3.5 Plus** | 7,460 次 | 2,460 次 | — | — |
| **Local LLMs** | 无限制 | 3,660 次 | 5,000 次 | 1,290 次 |

### 每次发言成本

#### 推荐模型

| LLM \ ASR | Qwen ASR (本地) | Qwen ASR (云端) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | ~0.0005 元 | ~0.002 元 | ~0.002 元 | ~0.006 元 |
| **DeepSeek V4 Flash** | ~0.0004 元 | ~0.002 元 | ~0.002 元 | ~0.006 元 |

#### 其他模型

| LLM \ ASR | Qwen ASR (本地) | Qwen ASR (云端) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 31B (Cerebras)** | ~0.008 元 | ~0.010 元 | ~0.009 元 | ~0.013 元 |
| **DeepSeek V4 Pro** | ~0.001 元 | ~0.003 元 | ~0.003 元 | ~0.007 元 |
| **Gemini 3 Flash** | ~0.004 元 | ~0.006 元 | ~0.006 元 | ~0.010 元 |
| **Gemini 3.1 Flash-Lite** | ~0.002 元 | ~0.004 元 | ~0.004 元 | ~0.008 元 |
| **Qwen 3.5 Plus** | ~0.001 元 | ~0.003 元 | — | — |
| **Local LLMs** | 0 元 | ~0.002 元 | ~0.001 元 | ~0.006 元 |

*   *(假设输入 900 token + 输出 12 token) × 每次发言平均 LLM 调用次数 1.2 次*
*   *每 1 美元可用次数以「每次发言成本」表中四舍五入前的计算值为准*
*   *所有费用与可用次数均为近似计算*
*   *DeepSeek 假设缓存命中率为 70%*
*   *Qwen API 计费以北京区域为准*
*   *资费标准截至：2026 年 5 月 25 日 / 启用快速响应模式*
*   *1 美元 ≈ 7.2 元人民币*

### 免费额度

| 服务 | 免费额度 | 期限 | 备注 |
|--------|------------|------|------|
| **Deepgram** | $200 | 无限制 | - |
| **Google AI Studio** | $10 | 1 年 | 面向 Gemini 订阅者每月发放 |
| **阿里云** | 每个模型 100 万 token | 90 天 | 新加坡区域为准 |
| **阿里云** | ¥300 | 1 年 | 面向中国境内学生 |
| **Cerebras** | 每天 100 万 token | 无限制 | 每分钟调用限制 5 次 |

---

# 如果遇到问题或有不明确的地方，欢迎随时通过 [Twitter/X](https://x.com/kapitalismho) 发 DM 联系我。

## 使用方法

1. 从[下载页面](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)下载最新版本
2. 安装 PuriPuly
3. 点击 **STT** 按钮
4. 点击 **TRANS** 按钮后通过 Discord 验证
5. 点击 **Subtitles** 按钮开启 VR 字幕
6. （可选）点击 **Peer** 按钮开启对方语音翻译

   > 对方语音翻译功能正常工作需要噪音较少的环境。若在 VRChat 中使用，请通过 Earmuff 功能控制环境。

7. 在 VRChat 中启用 OSC：Action menu → Settings → OSC → Enable

### 如果无法捕获音频
如果无法捕获音频，请在 **设置 > 常规** 中按以下步骤操作。

1. 将 **应用音频接口** 改为 **自动选择** 或 **MME**
2. 选择正确的麦克风
3. 重启应用

---

### 中国大陆用户指南

如果您所在地区无法访问 Soniox / Gemini / Deepgram，请使用以下组合。

- STT：**Qwen ASR**
- LLM：**DeepSeek V4 Flash**

   > 可以通过 QQ 认证，而不是 Discord。

---

### 使用您自己的 API 密钥

请根据您要使用的服务，参考对应指南操作。

翻译用 LLM 建议通过 OpenRouter 使用 Gemma 4 模型。

如果方便的话，既然要设置，何不顺便把 ASR 也一起配置好呢？
PuriPuly 与云端 STT 结合时能提供最佳体验。
例如即使同样是 Qwen ASR，本地与云端的语音识别性能也有相当差距。

<details>
<summary><h3>OpenRouter</h3></summary>

1. 请按下方截图所示设置红色圆圈内的选项。
   ![step0](docs/images/openrouter/0.png)

2. 在应用中点击红色圆圈内的按钮。
   ![step1](docs/images/openrouter/1.png)

3. 在 OpenRouter 中登录。
   ![step2](docs/images/openrouter/2.png)

4. 点击红色圆圈内的按钮以退出付款窗口。
   ![step3](docs/images/openrouter/3.png)

5. 点击 **Authorize** 按钮。
   ![step4](docs/images/openrouter/4.png)

6. 按需充值预付金。
   ![step5](docs/images/openrouter/5.png)

<details>
<summary><h3>点击 Authorize 后仍未完成认证</h3></summary>

如果点击 Authorize 后仍未通过认证，请重试，或按下方步骤手动申请 API 密钥并粘贴。

6. 点击右上角账户后，进入左侧的 API Keys 标签，再点击中央的 Create 按钮。
   ![step6](docs/images/openrouter/6.png)

7. 点击 Create 按钮。
   ![step7](docs/images/openrouter/7.png)

8. 点击按钮复制 API 密钥，然后粘贴到翻译器的 API 标签中。
   ![step8](docs/images/openrouter/8.png)

</details>

</details>

<details>
<summary><h3>DeepSeek</h3></summary>

1. 请按下方截图所示设置红色圆圈内的选项。
   ![step0](docs/images/deepseek/0.png)

2. 访问 [DeepSeek 官网](https://www.deepseek.com/en/) 并点击 **Access API** 按钮。
   ![step1](docs/images/deepseek/1.png)

3. 在主页登录。
   ![step2](docs/images/deepseek/2.png)

4. 切换到 API Keys 标签后，点击 **Create new API Keys**。
   ![step3](docs/images/deepseek/3.png)

5. 点击按钮复制 API 密钥后，粘贴到翻译器的 API 标签中。
   ![step4](docs/images/deepseek/4.png)

6. 切换到 Top Up 标签，按需充值预付金。
   ![step5](docs/images/deepseek/5.png)

</details>

<details>
<summary><h3>Deepgram</h3></summary>

1. 访问并登录 [Deepgram Console](https://console.deepgram.com/)。
   ![step1](docs/images/deepgram/1.png)

2. 当出现欢迎信息或调查问卷时，请点击 **Skip** 跳过。
   ![step2](docs/images/deepgram/2.png)

3. 在服务选择界面中，选择 **STT (Speech-to-Text)**。
   ![step3](docs/images/deepgram/3.png)

4. 在 API Keys 菜单中，点击 **Create a New API Key**。
   ![step4](docs/images/deepgram/4.png)

5. 输入密钥名称（例如：`puripuly`）并生成。
   ![step5](docs/images/deepgram/5.png)

6. 复制生成的密钥并粘贴到 PuriPuly 设置中。
   ![step6](docs/images/deepgram/6.png)

</details>

<details>
<summary><h3>Gemini</h3></summary>

1. 访问 [Google AI Studio](https://aistudio.google.com/apikey) 并点击 **Get API key** 按钮。
   ![step1](docs/images/gemini/1.png)

2. 创建一个新项目。
   ![step2](docs/images/gemini/2.png)

3. 随意起一个名字。
   ![step3](docs/images/gemini/3.png)

4. 选择创建好的项目并点击 **Create key**。
   ![step4](docs/images/gemini/4.png)

5. 点击圆圈标记的地方。
   ![step5](docs/images/gemini/5.png)

6. 点击圆圈标记的地方并复制密钥。
   ![step6](docs/images/gemini/6.png)

7. （推荐）点击黄色高亮的 **Set Up Billing** 按钮，升级并切换到付费方案。
切换方案可能需要一些时间。
   ![step7](docs/images/gemini/7.png)

<details>
<summary><h3>Gemini 付费订阅用户</h3></summary>

8. 前往 [Google Developer Program](https://developers.google.com/program/my-benefits) 并加入该计划。
   ![step8](docs/images/gemini/8.png)

9. 选择您在第 7 步中设置的付费方案项目。
   ![step9](docs/images/gemini/9.png)

</details>

</details>

<details>
<summary><h3>Qwen</h3></summary>

1. 根据您所在的地区，选择合适的链接访问阿里云百炼平台：
   - [中国大陆](https://bailian.console.aliyun.com/cn-beijing)
   - [中国大陆以外的地区](https://bailian.console.alibabacloud.com)

2. 在访问的地址中登录。请准确选择您希望申请 API 密钥的区域（Region）。（例如：Beijing）
   ![step2](docs/images/qwen/1.png)

3. 点击右上角的 **齿轮图标**。
   ![step3](docs/images/qwen/2.png)

4. 创建一个工作空间（Workspace），然后进入 **API-KEY** 页面。
   ![step4](docs/images/qwen/3.png)

5. 点击 **Create API Key**。
   ![step5](docs/images/qwen/4.png)

6. 分配账号和工作空间，然后点击 OK 按钮。
   ![step6](docs/images/qwen/5.png)

7. 点击圆圈标记的地方以复制密钥。
   ![step7](docs/images/qwen/6.png)

</details>

<details>
<summary><h3>Soniox</h3></summary>

1. 登录 [Soniox Console](https://console.soniox.com/)。
   ![step1](docs/images/soniox/1.png)

2. 随意填写一个组织名称。
   ![step2](docs/images/soniox/2.png)

3. 点击 **Add Funds** 按钮以绑定支付方式。
   ![step3](docs/images/soniox/3.png)

4. Soniox 需要预先充值余额。充值完成后，请前往 **API Keys** 菜单。
   ![step4](docs/images/soniox/4.png)

5. 创建一个新的 API Key。
   ![step5](docs/images/soniox/5.png)

6. 复制生成的密钥并粘贴到 PuriPuly 设置中。
   ![step6](docs/images/soniox/6.png)

</details>

<details>
<summary><h3>Cerebras</h3></summary>

1. 访问 [Cerebras](https://www.cerebras.ai/) 并点击 **Get started** 按钮。
   ![step1](docs/images/cerebras/1.png)

2. 登录。
   ![step2](docs/images/cerebras/2.png)

3. 选择想要的方案。首次使用建议选择免费层级。
   ![step3](docs/images/cerebras/3.png)

4. 复制 API 密钥并粘贴到 PuriPuly。
   ![step4](docs/images/cerebras/4.png)

<details>
<summary><h3>切换到付费层级</h3></summary>

5. 前往 **Billing** 标签。
   ![step5](docs/images/cerebras/5.png)

6. 填写您的姓名。
   ![step6](docs/images/cerebras/6.png)

7. 按需充值额度。
   ![step7](docs/images/cerebras/7.png)

</details>

</details>

---

## 开发

### 开发环境概览

| 领域 | 推荐环境 |
|---|---|
| Python 应用 | Windows |
| VR 浮层 | Windows |
| Broker 服务 | Linux / WSL |

### Python 应用

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

```bash
# pip
pip install -e '.[dev]'

# 或者 uv
uv sync --dev
```

```bash
pre-commit install
```

### 运行 GUI

```bash
# 激活虚拟环境后
python -m puripuly_heart.main run-gui

# 或通过 uv 运行
uv run python -m puripuly_heart.main run-gui
```

```bash
# 可查看隐藏的 UI
python -m puripuly_heart.main run-gui --debug-ui-preview
```

### 测试与代码检查

```bash
black src tests          # 格式化
ruff check src tests     # 代码检查
python -m pytest         # 测试 (建议在虚拟环境中运行)
```

### VR 浮层

VR 字幕浮层在 `native/overlay/` 的 Rust 项目中构建。

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

### Broker 服务

详情请参阅 `broker/README.md`。

```bash
pnpm install --frozen-lockfile
pnpm run typecheck
pnpm exec vitest run
pnpm --filter @puripuly-heart/broker run verify:config
pnpm --filter @puripuly-heart/broker run dev
```

---

## 开发者

[salee](https://github.com/kapitalismho)

---

## 贡献者

[RICHARDwuxiaofei](https://github.com/RICHARDwuxiaofei)

---

## Special Thanks

SUI\_32C, Nagikokoro, motoka96, \_Ykol魚, kascr\_, Just Monika V, FLUVIA, Han โชเล่ย์, EA\_PE, Ephedrine, ~ eri ~

---

## 许可证

[AGPL-3.0-or-later](LICENSE)

第三方许可证与声明：`src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`
