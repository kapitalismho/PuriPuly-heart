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

## Utveckling

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

## Utvecklare
[salee](https://github.com/kapitalismho)

## Licens
[AGPL-3.0-or-later](LICENSE)
