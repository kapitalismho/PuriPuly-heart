# PuriPuly Heart Portable — i18n & README Expansion Documentation

**Date:** 2026-07-10  
**Project:** PuriPuly-heart-portable-v1 / fzcfweasdferttgg-png/PuriPuly-heart  
**Branch:** `portable`

---

## Summary

### What was done

1. **i18n JSON files** — expanded from 4 to 35 UI locales
2. **README translations** — completed 31 incomplete translations + generated Korean from stub
3. **Portable release** — updated v1.0.0-portable with all new content

---

## i18n JSON Files

**Location:** `app/src/puripuly_heart/data/i18n/`

| Category | Count | Keys |
|----------|-------|------|
| Reference (en.json) | 1 | 531 |
| Pre-existing | 10 | 531 |
| New files | 24 | 534 |

**35 locale files:** en, ar, bg, ca, cs, da, de, el, es, et, fi, fr, hi, hu, id, it, ja, ko, lt, lv, ms, nl, no, pl, pt, ro, ru, sk, sv, th, tr, uk, vi, zh-CN, zh-TW

**Extra keys in new files (harmless, unused by app):**
- `openrouter.handoff.title`
- `peer_translation_eula.show_again`
- `peer_translation_eula.show_again.description`

**Added to all non-en files:** `peer_translation.disclosure` key

---

## README Translations

**Location:** `app/` (root of portable fork)

| Language | File | Status |
|----------|------|--------|
| English | README.md | Reference (547 lines) |
| Arabic | README.ar.md | ✅ Complete |
| Bulgarian | README.bg.md | ✅ Complete |
| Catalan | README.ca.md | ✅ Complete |
| Czech | README.cs.md | ✅ Complete |
| Danish | README.da.md | ✅ Complete |
| German | README.de.md | ✅ Complete |
| Greek | README.el.md | ✅ Complete |
| Spanish | README.es.md | ✅ Complete |
| Estonian | README.et.md | ✅ Complete |
| Finnish | README.fi.md | ✅ Complete |
| French | README.fr.md | ✅ Complete |
| Hindi | README.hi.md | ✅ Complete |
| Hungarian | README.hu.md | ✅ Complete |
| Indonesian | README.id.md | ✅ Complete |
| Italian | README.it.md | ✅ Complete |
| Japanese | README.ja.md | ✅ Complete |
| Korean | README.ko.md | ✅ Generated from stub |
| Lithuanian | README.lt.md | ✅ Complete |
| Latvian | README.lv.md | ✅ Complete |
| Malay | README.ms.md | ✅ Complete |
| Dutch | README.nl.md | ✅ Complete |
| Norwegian | README.no.md | ✅ Complete |
| Polish | README.pl.md | ✅ Complete |
| Portuguese | README.pt.md | ✅ Complete |
| Romanian | README.ro.md | ✅ Complete |
| Russian | README.ru.md | ✅ Complete |
| Slovak | README.sk.md | ✅ Complete |
| Swedish | README.sv.md | ✅ Complete |
| Thai | README.th.md | ✅ Complete |
| Turkish | README.tr.md | ✅ Complete |
| Ukrainian | README.uk.md | ✅ Complete |
| Vietnamese | README.vi.md | ✅ Complete |
| Chinese Simplified | README.zh-CN.md | ✅ Complete |
| Chinese Traditional | README.zh-TW.md | ✅ Complete |

**Sections added to incomplete files:**
- Audio capture troubleshooting
- China users note
- API key setup guides (OpenRouter, DeepSeek, Deepgram, Gemini, Qwen, Soniox, Cerebras)
- Environment Summary table
- Python App setup
- Running the GUI
- Testing & Linting
- VR Overlay build
- Broker Service
- Contributors
- Special Thanks

---

## GitHub Commits

| Commit | Description | Files |
|--------|-------------|-------|
| `2219746` | i18n JSON expansion (4 → 35 locales) | 35 |
| `e072670` | README translations completion | 31 |

**Repository:** https://github.com/fzcfweasdferttgg-png/PuriPuly-heart  
**Branch:** `portable`

---

## Release

**Tag:** `v1.0.0-portable`  
**Archive:** `PuriPulyHeart-Portable-v1.0.7z` (900 MB)  
**URL:** https://github.com/fzcfweasdferttgg-png/PuriPuly-heart/releases/tag/v1.0.0-portable

**Release contents:**
- Embedded Python 3.12
- All dependencies
- Qwen3 ASR model (~987 MB)
- 35 UI locale translations
- 35 README translations

---

## Directory Structure

```
PuriPuly-heart-portable-v1/
├── app/
│   ├── README.md (English - 547 lines)
│   ├── README.ar.md ... README.zh-TW.md (34 translations)
│   └── src/
│       └── puripuly_heart/
│           └── data/
│               └── i18n/
│                   ├── en.json (531 keys - reference)
│                   ├── ar.json ... vi.json (34 files)
│                   └── zh-TW.json
├── data/
├── mod/
├── python/
├── start.exe
├── start-debug.exe
└── start.cs
```

---

## How to Update in Future

### Adding new locale
1. Create `app/src/puripuly_heart/data/i18n/{code}.json` based on `en.json`
2. Create `app/README.{code}.md` based on `README.md`
3. Add locale entry to `en.json` `locale.*` keys
4. Update `i18n.py` `_LOCALE_DISPLAY_ORDER`
5. Update `settings.py` `resolve_first_run_ui_locale`
6. Commit, push, update release

### Updating translations
1. Edit the JSON/README file directly
2. Commit and push
3. Rebuild7z archive
4. Update release asset

---

## Notes

1. **Translation quality:** JSON translations done by AI (MiMo-V2.5-Pro) — may need human review
2. **API guide screenshots:** Identical across all languages (English screenshots)
3. **Extra keys:** 3 extra keys in new JSON files are harmless
4. **Korean README:** Generated from scratch (was a stub)

---

**Status:** ✅ ALL COMPLETE  
**Last updated:** 2026-07-10
