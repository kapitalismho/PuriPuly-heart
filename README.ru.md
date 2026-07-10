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

<p align="center">Двусторонний переводчик для VRChat на базе LLM</p>

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
  🇷🇺 Русский ·
  <a href="README.sk.md">🇸🇰 Slovenčina</a> ·
  <a href="README.sv.md">🇸🇪 Svenska</a> ·
  <a href="README.th.md">🇹🇭 ไทย</a> ·
  <a href="README.tr.md">🇹🇷 Türkçe</a> ·
  <a href="README.uk.md">🇺🇦 Українська</a> ·
  <a href="README.vi.md">🇻🇳 Tiếng Việt</a> ·
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README.zh-TW.md">🇹🇼 繁體中文</a>
</h2>

> ⚠️ **Это портативный форк** [kapitalismho/PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart). Изменён для удобного распространения и модификации. [Скачать портативную сборку →](../../releases)

---

## Демо

![Сравнение результатов перевода PuriPuly (Deepgram + Gemini 3 Flash) и VRCT (Google Web Speech + Google Translate).](docs/images/demo/ko-en_screenshot.png)

---

<video src="https://github.com/user-attachments/assets/c667f44d-b91d-42a9-b24a-e6a993b392d3" controls width="100%"></video>

Больше примеров реального общения с иностранцами через PuriPuly:
- [Демо 1](https://www.youtube.com/watch?v=3p0CamYui0o)
- [Демо 2](https://youtu.be/DoX36Y7J_lc?si=YjbeVTS8v3jGQB1w)
- [Демо 3](https://www.youtube.com/watch?v=D0npvp68xNY)

---

## Наконец-то говорите как настоящие друзья.

Вы были в такой ситуации.
Хотели поддержать друга,
но получилось только: «Ты в порядке?»

Вы и так знаете, что «переводчик»
не способен передать то, что на сердце.

Поэтому я создал такой, который может.

- **Перевод на базе LLM** — сленг, разговорные выражения, формальная и неформальная речь — всё передаётся естественно.
- **Память контекста** — разговор течёт自然而 с учётом предыдущего контекста.
- **Двусторонний голосовой перевод** — переводит также речь собеседника, с поддержкой субтитров в VR.
- **Старт через Discord** — начните использовать сразу, без сложной настройки.

## Вопросы и ответы

- **Насколько хорош качество перевода?**
→ Когда оба собеседника используют PuriPuly, можно вести даже самые глубокие разговоры. Количественно, с Gemma 4 результат в 6 раз лучше, чем у DeepL. Подробности в разделе «Сравнение перевода» ниже.

- **Сколько времени проходит от произнесения фразы до получения перевода?**
→ С Gemma 4 и облачным STT задержка обычно составляет около полутора секунд.

- **Это бесплатно?**
→ Да, но не сразу. Новые пользователи получают бесплатный лимит, а после его исчерпания цены очень низкие — тысячи фраз за $1.

- **Нужен ли API-ключ?**
→ Да, но тоже не сразу. Просто установите и авторизуйтесь через Discord — и можно пользоваться.

- **Как работает перевод речи собеседника?**
→ Лучше всего работает в разговорах один на один в тихой обстановке. До трёх человек тоже может работать, но не гарантировано. В VRChat используйте Earmuff для контроля окружения.

- **Распознавание работает плохо / медленно.**
→ Если вы используете локальный Qwen ASR, рекомендуем переключиться на облачный STT. На процессорах Intel настройте PuriPuly на использование только производительных ядер (P-cores).

- **Как обрабатываются голос и содержание разговора?**
→ Голос и содержание разговора хранятся локально и не отправляются на серверы PuriPuly. Голоса других людей, транскрипции и результаты перевода никогда не записываются. Тем не менее, STT-сервис и провайдер перевода могут обрабатывать эти данные.

### [📥 Скачать](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)

---

## Сравнение перевода
![График качества перевода. Средний штраф за ошибки на предложение (чем ниже, тем лучше) по методологии Gemba MQM (модель-судья: Gemini 3.1 Pro Preview) на 216 многоходовых примерах KO → EN, JA, ZH-Hans.](docs/images/performance/1.png)

- Эксперимент проведён с использованием методологии Microsoft Gemba MQM.
- Настройка многоходовая, для максимального приближения к реальному разговору.
- Полные результаты [здесь](https://github.com/kapitalismho/korean-llm-context-translation-benchmark).

## Стоимость

### Использований за доллар

#### Рекомендуемые модели

| LLM \ ASR | Qwen ASR (локальный) | Qwen ASR (облачный) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | 14,380 | 2,920 | 3,710 | 1,180 |
| **DeepSeek V4 Flash** | 19,410 | 3,080 | 3,980 | 1,210 |

#### Другие модели

| LLM \ ASR | Qwen ASR (локальный) | Qwen ASR (облачный) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 31B (Cerebras)** | 920 | 730 | 770 | 540 |
| **DeepSeek V4 Pro** | 6,400 | 2,330 | 2,810 | 1,070 |
| **Gemini 3 Flash** | 1,710 | 1,170 | 1,280 | 740 |
| **Gemini 3.1 Flash-Lite** | 3,430 | 1,770 | 2,030 | 940 |
| **Qwen 3.5 Plus** | 7,460 | 2,460 | — | — |
| **Локальные LLM** | Без ограничений | 3,660 | 5,000 | 1,290 |

### Стоимость одной фразы

#### Рекомендуемые модели

| LLM \ ASR | Qwen ASR (локальный) | Qwen ASR (облачный) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | ~$0.00007 | ~$0.0003 | ~$0.0003 | ~$0.0008 |
| **DeepSeek V4 Flash** | ~$0.00005 | ~$0.0003 | ~$0.0003 | ~$0.0008 |

#### Другие модели

| LLM \ ASR | Qwen ASR (локальный) | Qwen ASR (облачный) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 31B (Cerebras)** | ~$0.0011 | ~$0.0014 | ~$0.0013 | ~$0.0019 |
| **DeepSeek V4 Pro** | ~$0.0002 | ~$0.0004 | ~$0.0004 | ~$0.0009 |
| **Gemini 3 Flash** | ~$0.0006 | ~$0.0009 | ~$0.0008 | ~$0.0014 |
| **Gemini 3.1 Flash-Lite** | ~$0.0003 | ~$0.0006 | ~$0.0005 | ~$0.0011 |
| **Qwen 3.5 Plus** | ~$0.0001 | ~$0.0004 | — | — |
| **Локальные LLM** | $0 | ~$0.0003 | ~$0.0002 | ~$0.0008 |

*   *Расчёт: (Вход 900 токенов + Выход 12 токенов) × 1.2 вызовов LLM на фразу.*
*   *Все стоимости и количества приблизительные.*
*   *DeepSeek предполагает 70% попаданий в кэш.*
*   *Стоимость Qwen API — для региона Пекин.*
*   *Цены на 25 мая 2026 / режим быстрого ответа.*

### Бесплатные кредиты

| Сервис | Бесплатный кредит | Срок | Примечание |
|--------|------------|------|------|
| **Deepgram** | $200 | Без ограничений | - |
| **Google AI Studio** | $10 | 1 год | Ежемесячно для подписчиков Gemini |
| **Alibaba Cloud** | 1M токенов на модель | 90 дней | Регион Сингапур |
| **Alibaba Cloud** | ¥300 | 1 год | Студенты в Китае |
| **Cerebras** | 1M токенов ежедневно | Без ограничений | Лимит 5 вызовов в минуту |

---

# Если возникнут вопросы или что-то непонятно, пишите в [Twitter/X](https://x.com/kapitalismho).

## Использование

1. Скачайте последнюю версию со [страницы загрузки](https://github.com/kapitalismho/PuriPuly-heart/releases/latest).
2. Установите PuriPuly.
3. Нажмите кнопку **STT**.
4. Нажмите кнопку **TRANS**, затем авторизуйтесь через Discord.
5. Нажмите кнопку **Subtitles** для включения VR-субтитров.
6. (Опционально) Нажмите кнопку **Peer** для включения перевода речи собеседника.

   > Для перевода речи собеседника нужна тихая обстановка. В VRChat используйте Earmuff.

7. Включите OSC в VRChat: Меню действий → Настройки → OSC → Включить.

### Если захват звука не работает
Откройте **Настройки > Основные** и выполните:

1. Измените **Audio Host API** на **Auto** или **MME**.
2. Выберите правильный микрофон.
3. Перезапустите приложение.

---

### Примечание для пользователей из Китая

Если Soniox/Gemini/Deepgram заблокированы в вашем регионе, используйте:

- STT: **Qwen ASR**
- LLM: **DeepSeek V4 Flash**

   > Авторизация через QQ вместо Discord.

---

### Использование собственных API-ключей

Следуйте инструкции для нужного сервиса.

Для перевода рекомендуем модель Gemma 4 через OpenRouter.

Заодно настройте ASR — PuriPuly работает лучше всего с облачным STT.
Даже с одним и тем же Qwen ASR разница между локальным и облачным распознаванием заметна.

Рекомендуем начать с Deepgram — при регистрации дают $200 бесплатных кредитов.

<details>
<summary><h3>OpenRouter</h3></summary>

1. Установите параметры как на скриншоте.
   ![step0](docs/images/openrouter/0.png)

2. В приложении нажмите кнопку в красном круге.
   ![step1](docs/images/openrouter/1.png)

3. Войдите в OpenRouter.
   ![step2](docs/images/openrouter/2.png)

4. Нажмите кнопку чтобы выйти из экрана оплаты.
   ![step3](docs/images/openrouter/3.png)

5. Нажмите кнопку **Authorize**.
   ![step4](docs/images/openrouter/4.png)

6. Пополните баланс на нужную сумму.
   ![step5](docs/images/openrouter/5.png)

<details>
<summary><h3>Если кнопка Authorize не сработала</h3></summary>

Попробуйте ещё раз или создайте API-ключ вручную:

6. Нажмите на аккаунт в правом верхнем углу, перейдите в API Keys, затем нажмите Create.
   ![step6](docs/images/openrouter/6.png)

7. Нажмите Create.
   ![step7](docs/images/openrouter/7.png)

8. Скопируйте ключ и вставьте его в раздел API переводчика.
   ![step8](docs/images/openrouter/8.png)

</details>

</details>

<details>
<summary><h3>DeepSeek</h3></summary>

1. Установите параметры как на скриншоте.
   ![step0](docs/images/deepseek/0.png)

2. Перейдите на [сайт DeepSeek](https://www.deepseek.com/en/) и нажмите **Access API**.
   ![step1](docs/images/deepseek/1.png)

3. Войдите на сайт.
   ![step2](docs/images/deepseek/2.png)

4. Перейдите в API Keys и нажмите **Create new API Keys**.
   ![step3](docs/images/deepseek/3.png)

5. Скопируйте ключ и вставьте в раздел API переводчика.
   ![step4](docs/images/deepseek/4.png)

6. Перейдите в Top Up и пополните баланс.
   ![step5](docs/images/deepseek/5.png)

</details>

<details>
<summary><h3>Deepgram</h3></summary>

1. Войдите в [Deepgram Console](https://console.deepgram.com/).
   ![step1](docs/images/deepgram/1.png)

2. Если видите приветствие/опрос, нажмите **Skip**.
   ![step2](docs/images/deepgram/2.png)

3. Выберите **STT (Speech-to-Text)**.
   ![step3](docs/images/deepgram/3.png)

4. В меню API Keys нажмите **Create a New API Key**.
   ![step4](docs/images/deepgram/4.png)

5. Введите имя ключа (например `puripuly`) и создайте.
   ![step5](docs/images/deepgram/5.png)

6. Скопируйте ключ и вставьте в настройки PuriPuly.
   ![step6](docs/images/deepgram/6.png)

</details>

<details>
<summary><h3>Gemini</h3></summary>

1. Перейдите в [Google AI Studio](https://aistudio.google.com/apikey) и нажмите **Get API key**.
   ![step1](docs/images/gemini/1.png)

2. Создайте новый проект.
   ![step2](docs/images/gemini/2.png)

3. Выберите любое имя для проекта.
   ![step3](docs/images/gemini/3.png)

4. Выберите созданный проект и нажмите **Create key**.
   ![step4](docs/images/gemini/4.png)

5. Нажмите на выделенную область.
   ![step5](docs/images/gemini/5.png)

6. Скопируйте ключ.
   ![step6](docs/images/gemini/6.png)

7. (Рекомендуется) Нажмите жёлтую кнопку **Set Up Billing** для перехода на платный тариф.
   ![step7](docs/images/gemini/7.png)

<details>
<summary><h3>Для платных подписчиков Gemini</h3></summary>

8. Перейдите в [Google Developer Program](https://developers.google.com/program/my-benefits) и присоединитесь.
   ![step8](docs/images/gemini/8.png)

9. Выберите проект с платным тарифом из шага 7.
   ![step9](docs/images/gemini/9.png)

</details>

</details>

<details>
<summary><h3>Qwen</h3></summary>

1. Откройте Alibaba Cloud Model Studio:
   - [Китай](https://bailian.console.aliyun.com/cn-beijing)
   - [За пределами Китая](https://bailian.console.alibabacloud.com)

2. Войдите. Выберите правильный регион для API-ключа (например Пекин).
   ![step2](docs/images/qwen/1.png)

3. Нажмите на **иконку шестерёнки** в правом верхнем углу.
   ![step3](docs/images/qwen/2.png)

4. Создайте рабочее пространство и перейдите на страницу **API-KEY**.
   ![step4](docs/images/qwen/3.png)

5. Нажмите **Create API Key**.
   ![step5](docs/images/qwen/4.png)

6. Выберите аккаунт и рабочее пространство, нажмите OK.
   ![step6](docs/images/qwen/5.png)

7. Скопируйте ключ.
   ![step7](docs/images/qwen/6.png)

</details>

<details>
<summary><h3>Soniox</h3></summary>

1. Войдите в [Soniox Console](https://console.soniox.com/).
   ![step1](docs/images/soniox/1.png)

2. Введите название организации.
   ![step2](docs/images/soniox/2.png)

3. Нажмите **Add Funds** для привязки способа оплаты.
   ![step3](docs/images/soniox/3.png)

4. Soniox требует предоплаты. После пополнения перейдите в **API Keys**.
   ![step4](docs/images/soniox/4.png)

5. Создайте новый API Key.
   ![step5](docs/images/soniox/5.png)

6. Скопируйте ключ и вставьте в настройки PuriPuly.
   ![step6](docs/images/soniox/6.png)

</details>

<details>
<summary><h3>Cerebras</h3></summary>

1. Перейдите на [Cerebras](https://www.cerebras.ai/) и нажмите **Get started**.
   ![step1](docs/images/cerebras/1.png)

2. Войдите.
   ![step2](docs/images/cerebras/2.png)

3. Выберите тариф. Рекомендуем начать с бесплатного.
   ![step3](docs/images/cerebras/3.png)

4. Скопируйте API-ключ и вставьте в PuriPuly.
   ![step4](docs/images/cerebras/4.png)

<details>
<summary><h3>Переход на платный тариф</h3></summary>

5. Перейдите на вкладку **Billing**.
   ![step5](docs/images/cerebras/5.png)

6. Введите имя.
   ![step6](docs/images/cerebras/6.png)

7. Пополните баланс на нужную сумму.
   ![step7](docs/images/cerebras/7.png)

</details>

</details>

---

## Разработка

### Среда разработки

| Область | Рекомендуемая среда |
|---|---|
| Python-приложение | Windows |
| VR-оверлей | Windows |
| Broker-сервис | Linux / WSL |

### Python-приложение

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

```bash
# pip
pip install -e '.[dev]'

# или uv
uv sync --dev
```

```bash
pre-commit install
```

### Запуск GUI

```bash
# После активации venv
python -m puripuly_heart.main run-gui

# или через uv
uv run python -m puripuly_heart.main run-gui
```

```bash
# Скрытый интерфейс для разработки
python -m puripuly_heart.main run-gui --debug-ui-preview
```

### Тесты и линтинг

```bash
black src tests          # Форматирование
ruff check src tests     # Линтинг
python -m pytest         # Тесты (рекомендуется в venv)
```

### VR-оверлей

VR-субтитры собираются из Rust-проекта в `native/overlay/`.

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

### Broker-сервис

См. `broker/README.md`.

```bash
pnpm install --frozen-lockfile
pnpm run typecheck
pnpm exec vitest run
pnpm --filter @puripuly-heart/broker run verify:config
pnpm --filter @puripuly-heart/broker run dev
```

---

## Разработчик

[salee](https://github.com/kapitalismho)

---

## Участники

[RICHARDwuxiaofei](https://github.com/RICHARDwuxiaofei)

---

## Благодарности

SUI\_32C, Nagikokoro, motoka96, \_Ykol魚, kascr\_, Just Monika V, FLUVIA, Han โชเล่ย์, EA\_PE, Ephedrine, ~ eri ~

---

## Лицензия

[AGPL-3.0-or-later](LICENSE)

Сторонние лицензии и уведомления: `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`
