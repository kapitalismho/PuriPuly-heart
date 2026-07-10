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

## Pengembangan

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

## Pengembang
[salee](https://github.com/kapitalismho)

## Lisensi
[AGPL-3.0-or-later](LICENSE)
