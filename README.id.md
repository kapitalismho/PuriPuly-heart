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

<p align="center">Penerjemah dua arah untuk VRChat dengan LLM</p>

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
  🇮🇩 Bahasa Indonesia ·
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
  <a href="README.zh-TW.md">🇹🇼 繁體中文</a>
</h2>

> ⚠️ **Ini adalah fork portabel dari** [kapitalismho/PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart). Dimodifikasi untuk distribusi dan modifikasi yang mudah. [Unduh versi portabel ←](../../releases)

---

## Demo

![Perbandingan hasil terjemahan antara PuriPuly dan VRCT.](docs/images/demo/ko-en_screenshot.png)

---

<video src="https://github.com/user-attachments/assets/c667f44d-b91d-42a9-b24a-e6a993b392d3" controls width="100%"></video>

Contoh komunikasi nyata melalui PuriPuly:
- [Demo 1](https://www.youtube.com/watch?v=3p0CamYui0o)
- [Demo 2](https://youtu.be/DoX36Y7J_lc?si=YjbeVTS8v3jGQB1w)
- [Demo 3](https://www.youtube.com/watch?v=D0npvp68xNY)

---

## Akhirnya, bicaralah seperti teman sejati.

Kamu pernah di situasi itu.
Ingin menghibur teman,
tapi hanya bisa: "Kamu baik-baik saja?"

Kamu tahu "penerjemah" tidak bisa menyampaikan apa yang ada di hatimu.
Makanya aku buat yang bisa.

- **Terjemahan dengan LLM** — slang, bahasa sehari-hari, formal dan informal — semua tersampaikan secara alami.
- **Memori konteks** — percakapan mengalir alami dengan kesadaran konteks sebelumnya.
- **Terjemahan suara dua arah** — juga menerjemahkan suara orang lain, dengan dukungan subtitle VR.
- **Mulai via Discord** — langsung mulai tanpa pengaturan rumit.

## Pertanyaan umum

- **Bagaimana kualitas terjemahannya?**
→ Dengan Gemma 4, hasilnya 6x lebih baik dari DeepL.

- **Berapa lama waktunya?**
→ Dengan Gemma 4 dan cloud STT, latensi sekitar satu setengah detik.

- **Apakah berbayar?**
→ Ya, tapi tidak langsung. Pengguna baru mendapat kredit gratis. Setelah itu, harga sangat murah — ribuan kalimat seharga $1.

- **Apakah butuh API key?**
→ Ya, tapi tidak langsung. Cukup instal dan autentikasi via Discord.

- **Bagaimana terjemahan suara orang lain bekerja?**
→ Paling baik dalam percakapan satu lawan satu di lingkungan tenang. Di VRChat gunakan Earmuff.

- **Pengenalan suara buruk/lambat.**
→ Gunakan cloud STT. Di Intel, gunakan hanya inti performa.

- **Bagaimana data dikelola?**
→ Suara dan isi percakapan disimpan secara lokal dan tidak dikirim ke server.

### [📥 Unduh](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)

---

## Perbandingan terjemahan
![Grafik kualitas terjemahan via Gemba MQM.](docs/images/performance/1.png)

- Eksperimen dengan Microsoft Gemba MQM.
- Lingkungan multi-putaran untuk percakapan lebih realistis.
- Hasil lengkap [di sini](https://github.com/kapitalismho/korean-llm-context-translation-benchmark).

## Biaya

### Penggunaan per dolar

| LLM \ ASR | Qwen ASR (lokal) | Qwen ASR (cloud) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | 14,380 | 2,920 | 3,710 | 1,180 |
| **DeepSeek V4 Flash** | 19,410 | 3,080 | 3,980 | 1,210 |

### Kredit gratis

| Layanan | Kredit gratis | Durasi | Catatan |
|--------|------------|------|------|
| **Deepgram** | $200 | Tanpa batas | - |
| **Google AI Studio** | $10 | 1 tahun | Bulanan untuk pelanggan Gemini |
| **Alibaba Cloud** | 1M token per model | 90 hari | Wilayah Singapura |
| **Cerebras** | 1M token harian | Tanpa batas | Batas 5 panggilan/menit |

---

## Penggunaan

1. Unduh dari [halaman unduh](https://github.com/kapitalismho/PuriPuly-heart/releases/latest).
2. Instal PuriPuly.
3. Tekan **STT**.
4. Tekan **TRANS** dan autentikasi via Discord.
5. Tekan **Subtitles** untuk subtitle VR.
6. Tekan **Peer** untuk terjemahan suara orang lain.
7. Aktifkan OSC di VRChat: Menu ← Pengaturan ← OSC ← Aktifkan.

---

---

### Jika perekaman audio tidak berfungsi
Jika perekaman audio tidak berfungsi, buka **Pengaturan > Umum** dan ikuti langkah-langkah berikut.

1. Ubah **Audio Host API** ke **Auto** atau **MME**.
2. Pilih mikrofon yang benar.
3. Mulai ulang aplikasi.

---

### Catatan untuk pengguna di Tiongkok

Jika Soniox/Gemini/Deepgram diblokir di wilayah Anda, gunakan kombinasi berikut:

- STT: **Qwen ASR**
- LLM: **DeepSeek V4 Flash**

   > Anda dapat melakukan autentikasi melalui QQ alih-alih Discord.

---

### Menggunakan kunci API Anda sendiri

Ikuti panduan untuk layanan yang ingin Anda gunakan.

Untuk penerjemahan, kami merekomendasikan model Gemma 4 melalui OpenRouter.

Sekaligus konfigurasikan ASR — PuriPuly memberikan pengalaman terbaik dengan STT cloud.
Bahkan dengan Qwen ASR yang sama, perbedaan antara pengenalan lokal dan cloud terlihat jelas.

Kami merekomendasikan untuk memulai dengan Deepgram.
Cukup mendaftar saja sudah mendapat $200 kredit gratis.

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
<summary><h3>Jika tombol Authorize tidak berhasil</h3></summary>

Jika Anda mengklik Authorize tetapi masih belum terautentikasi, coba lagi, atau buat kunci API secara langsung:

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
<summary><h3>Untuk pelanggan berbayar Gemini</h3></summary>

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
<summary><h3>Beralih ke paket berbayar</h3></summary>

5. Go to the **Billing** tab.
   ![step5](docs/images/cerebras/5.png)

6. Enter your name.
   ![step6](docs/images/cerebras/6.png)

7. Add as much credit as you need.
   ![step7](docs/images/cerebras/7.png)

</details>

</details>

---

## Pengembangan

### Ringkasan lingkungan

| Area | Lingkungan yang direkomendasikan |
|---|---|
| Aplikasi Python | Windows |
| Overlay VR | Windows |
| Layanan broker | Linux / WSL |

### Aplikasi Python

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

### Menjalankan GUI

```bash
# After activating venv
python -m puripuly_heart.main run-gui

# or run via uv
uv run python -m puripuly_heart.main run-gui
```

```bash
# Menampilkan UI tersembunyi untuk inspeksi
python -m puripuly_heart.main run-gui --debug-ui-preview
```

### Pengujian dan linting

```bash
black src tests          # Pemformatan
ruff check src tests     # Linting
python -m pytest         # Pengujian (disarankan dalam venv)
```

### Overlay VR

Overlay subtitle VR dibangun dari proyek Rust di `native/overlay/`.

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

### Layanan broker

Lihat `broker/README.md`.

```bash
pnpm install --frozen-lockfile
pnpm run typecheck
pnpm exec vitest run
pnpm --filter @puripuly-heart/broker run verify:config
pnpm --filter @puripuly-heart/broker run dev
```

---

## Pengembang

[salee](https://github.com/kapitalismho)

---

## Kontributor

[RICHARDwuxiaofei](https://github.com/RICHARDwuxiaofei)

---

## Terima kasih khusus

SUI\_32C, Nagikokoro, motoka96, \_Ykol魚, kascr\_, Just Monika V, FLUVIA, Han โชเล่ย์, EA\_PE, Ephedrine, ~ eri ~

---

## Lisensi

[AGPL-3.0-or-later](LICENSE)

Lisensi dan pemberitahuan pihak ketiga: `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`
