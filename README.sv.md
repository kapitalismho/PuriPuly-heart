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

<p align="center">Tvåvägs översättare för VRChat driven av LLM</p>

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
  🇸🇪 Svenska ·
  <a href="README.th.md">🇹🇭 ไทย</a> ·
  <a href="README.tr.md">🇹🇷 Türkçe</a> ·
  <a href="README.uk.md">🇺🇦 Українська</a> ·
  <a href="README.vi.md">🇻🇳 Tiếng Việt</a> ·
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README.zh-TW.md">🇹🇼 繁體中文</a>
</h2>

> ⚠️ **Detta är en bärbar fork av** [kapitalismho/PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart). Modifierad för enkel distribution och modifiering. [Ladda ner bärbar version ←](../../releases)

---

## Demo

![Jämförelse av översättningsresultat mellan PuriPuly och VRCT.](docs/images/demo/ko-en_screenshot.png)

---

<video src="https://github.com/user-attachments/assets/c667f44d-b91d-42a9-b24a-e6a993b392d3" controls width="100%"></video>

Fler exempel på verklig kommunikation genom PuriPuly:
- [Demo 1](https://www.youtube.com/watch?v=3p0CamYui0o)
- [Demo 2](https://youtu.be/DoX36Y7J_lc?si=YjbeVTS8v3jGQB1w)
- [Demo 3](https://www.youtube.com/watch?v=D0npvp68xNY)

---

## Äntligen, prata som riktiga vänner.

Du har varit i den situationen.
Ville trösta en vän,
men lyckades bara: "Mår du bra?"

Du vet redan att en "översättare" inte kan förmedla det du har i hjärtat.
Därför byggde jag en som kan det.

- **LLM-driven lokalisering** — slang, talspråk, formellt och informellt tal — allt återgivet naturligt.
- **Kontextminne** — samtalet flyter naturligt med medvetenhet om tidigare kontext.
- **Tvåvägs röstöversättning** — översätter också den andra personens röst, med VR-undertextstöd.
- **Start via Discord** — börja direkt utan komplicerad inställning.

## Vanliga frågor

- **Hur bra är översättningskvaliteten?**
→ Med Gemma 4 är resultatet 6x bättre än DeepL.

- **Hur lång tid tar det?**
→ Med Gemma 4 och moln-STT är fördröjningen cirka en och en halv sekund.

- **Kostar det?**
→ Ja, men inte direkt. Nya användare får gratis kredit. Därefter är priserna väldigt låga — tusentals meningar för 1$.

- **Behöver jag en API-nyckel?**
→ Ja, men inte direkt. Installera och autentisera via Discord.

- **Hur fungerar översättning av den andres röst?**
→ Fungerar bäst i en-till-en-samtal i lugn miljö. I VRChat använd Earmuff.

- **Röstigenkänning är dålig/långsam.**
→ Använd moln-STT. På Intel, använd bara prestandakärnor.

- **Hur hanteras data?**
→ Röst och samtalsinnehåll lagras lokalt och skickas inte till servrar.

### [📥 Ladda ner](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)

---

## Översättningsjämförelse
![Diagram för översättningskvalitet via Gemba MQM.](docs/images/performance/1.png)

- Experiment med Microsoft Gemba MQM.
- Fleromgångsmiljö för mer realistiskt samtal.
- Fullständiga resultat [här](https://github.com/kapitalismho/korean-llm-context-translation-benchmark).

## Kostnad

### Användningar per dollar

| LLM \ ASR | Qwen ASR (lokal) | Qwen ASR (moln) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | 14,380 | 2,920 | 3,710 | 1,180 |
| **DeepSeek V4 Flash** | 19,410 | 3,080 | 3,980 | 1,210 |

### Gratis krediter

| Tjänst | Gratis kredit | Varighet | Notering |
|--------|------------|------|------|
| **Deepgram** | $200 | Obegränsat | - |
| **Google AI Studio** | $10 | 1 år | Månadsvis för Gemini-prenumeranter |
| **Alibaba Cloud** | 1M tokens per modell | 90 dagar | Singapore-region |
| **Cerebras** | 1M tokens dagligen | Obegränsat | 5 samtal/min gräns |

---

## Användning

1. Ladda ner från [nedladdningssidan](https://github.com/kapitalismho/PuriPuly-heart/releases/latest).
2. Installera PuriPuly.
3. Tryck **STT**.
4. Tryck **TRANS** och autentisera via Discord.
5. Tryck **Subtitles** för VR-undertexter.
6. Tryck **Peer** för översättning av den andres röst.
7. Aktivera OSC i VRChat: Meny ← Inställningar ← OSC ← Aktivera.

---

---

### Om ljudinspelning inte fungerar
Om ljudinspelning inte fungerar, öppna **Inställningar > Allmänt** och följ dessa steg.

1. Ändra **Audio Host API** till **Auto** eller **MME**.
2. Välj rätt mikrofon.
3. Starta om appen.

---

### Anteckning för användare i Kina

Om Soniox/Gemini/Deepgram är blockerade i din region, använd följande kombination:

- STT: **Qwen ASR**
- LLM: **DeepSeek V4 Flash**

   > Du kan autentisera via QQ istället för Discord.

---

### Använda dina egna API-nycklar

Följ guiden för den tjänst du vill använda.

För översättning rekommenderar vi Gemma 4-modellen via OpenRouter.

Konfigurera även ASR — PuriPuly ger bästa upplevelsen med moln-STT.
Även med samma Qwen ASR är skillnaden mellan lokal och molnigenkänning märkbar.

Vi rekommenderar att börja med Deepgram.
Bara registrering ger $200 gratis kredit.

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
<summary><h3>Om Authorize-knappen inte fungerade</h3></summary>

Om du klickade på Authorize men fortfarande inte är autentiserad, försök igen eller skapa en API-nyckel direkt:

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
<summary><h3>För betalande Gemini-prenumeranter</h3></summary>

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
<summary><h3>Byt till betalplan</h3></summary>

5. Go to the **Billing** tab.
   ![step5](docs/images/cerebras/5.png)

6. Enter your name.
   ![step6](docs/images/cerebras/6.png)

7. Add as much credit as you need.
   ![step7](docs/images/cerebras/7.png)

</details>

</details>

---

## Utveckling

### Miljööversikt

| Område | Rekommenderad miljö |
|---|---|
| Python-app | Windows |
| VR-överlägg | Windows |
| Broker-tjänst | Linux / WSL |

### Python-app

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

### Starta GUI

```bash
# After activating venv
python -m puripuly_heart.main run-gui

# or run via uv
uv run python -m puripuly_heart.main run-gui
```

```bash
# Visar dolt gränssnitt för inspektion
python -m puripuly_heart.main run-gui --debug-ui-preview
```

### Testning och linting

```bash
black src tests          # Formatering
ruff check src tests     # Linting
python -m pytest         # Tester (rekommenderat i venv)
```

### VR-överlägg

VR-undertextöverlägget byggs från Rust-projektet under `native/overlay/`.

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

### Broker-tjänst

Se `broker/README.md`.

```bash
pnpm install --frozen-lockfile
pnpm run typecheck
pnpm exec vitest run
pnpm --filter @puripuly-heart/broker run verify:config
pnpm --filter @puripuly-heart/broker run dev
```

---

## Utvecklare

[salee](https://github.com/kapitalismho)

---

## Bidragsgivare

[RICHARDwuxiaofei](https://github.com/RICHARDwuxiaofei)

---

## Speciellt tack

SUI\_32C, Nagikokoro, motoka96, \_Ykol魚, kascr\_, Just Monika V, FLUVIA, Han โชเล่ย์, EA\_PE, Ephedrine, ~ eri ~

---

## Licens

[AGPL-3.0-or-later](LICENSE)

Tredjepartslicenser och -meddelanden: `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`
