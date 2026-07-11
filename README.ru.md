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

<p align="center">Двусторонний переводчик на основе LLM для VRChat</p>

<h2 align="center">
  <a href="README.md">🇺🇸 English</a> ·
  <a href="README.ko.md">🇰🇷 한국어</a> ·
  <a href="README.ja.md">🇯🇵 日本語</a> ·
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> ·
  🇷🇺 Русский
</h2>

---

## Демо

![Сравнение результатов перевода между PuriPuly (Deepgram + Gemini 3 Flash) и VRCT (Google Web Speech + Google Translate). PuriPuly распознавание: «아역시혼자기대하면안된다니깐», перевод: «(See, I knew I shouldn't have gotten my hopes up.)» | VRCT распознавание: «아 역시 혼자 기대하면 안 된다니까», перевод: «Oh, I guess you shouldn't expect it alone.»](docs/images/demo/ko-en_screenshot.png)

---

<video src="https://github.com/user-attachments/assets/c667f44d-b91d-42a9-b24a-e6a993b392d3" controls width="100%"></video>

Если хотите увидеть больше примеров реального общения с друзьями из других стран через PuriPuly:
- [Демо 1](https://www.youtube.com/watch?v=3p0CamYui0o)
- [Демо 2](https://youtu.be/DoX36Y7J_lc?si=YjbeVTS8v3jGQB1w)
- [Демо 3](https://www.youtube.com/watch?v=D0npvp68xNY)

---

## Наконец-то говори как настоящий друг.

Бывало же.
Хочешь поддержать друга,
а получается только: «Ты в порядке?»

Ты и сам знаешь, что «переводчик»
не способен передать то, что на сердце.

Поэтому я создал такой, который может.

- **Локализация на основе LLM** — сленг, разговорная речь, формальное и неформальное общение — всё звучит естественно.
- **Контекстная память** — поддерживает естественный ход беседы, помня предыдущий контекст.
- **Двусторонний голосовой перевод** — также переводит речь собеседника, с поддержкой субтитров в VR.
- **Запуск через Discord** — начните использовать сразу без сложной настройки.

## Вопросы и ответы

- **Насколько хорошее качество перевода?**
→ Когда вы и ваш собеседник используете этот переводчик, вы можете вести даже самые глубокие разговоры. Количественно, с Gemma 4 результат в 6 раз лучше, чем у DeepL. Подробности — в разделе «Сравнение перевода» ниже.

- **Сколько времени от произнесения фразы до получения перевода?**
→ С Gemma 4 и облачным сервисом распознавания речи задержка обычно составляет около полутора секунд.

- **Использование стоит денег?**
→ Да, но не сразу. Новые пользователи получают бесплатный кредит, а после этого цены очень низкие — тысячи использований за $1.

- **Нужен ли API-ключ?**
→ Да, но не сразу. Просто установите и авторизуйтесь через Discord, чтобы начать пользоваться.

- **Насколько хорошо работает перевод речи собеседника?**
→ Лучше всего работает при разговоре один на один в тихой обстановке. До трёх человек тоже может работать, но не гарантировано. В VRChat используйте Earmuff для контроля окружения.

- **Распознавание речи работает плохо / медленно.**
→ Если вы используете локальный Qwen ASR, рекомендуем переключиться на облачный сервис. Если у вас процессор Intel, настройте PuriPuly на использование только производительных ядер (P-cores).

- **Как обрабатываются голос и содержание разговора?**
→ Голос и содержание разговора хранятся локально и не отправляются на серверы PuriPuly. Голоса других людей, транскрипции и результаты перевода никогда не записываются. Тем не менее, сервис распознавания речи и провайдер перевода могут обрабатывать эти данные.

### [📥 Скачать](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)

---

## Сравнение перевода

![Диаграмма качества перевода. Показано среднее штрафное значение ошибок на предложение (чем ниже — тем лучше), оценённое с помощью фреймворка Gemba MQM (модель-судья: Gemini 3.1 Pro Preview) на 216 многоходовых корейских образцах в парах EN, JA и ZH-Hans. Оценки: Gemini 3.1 Flash-lite 0.573, Gemini 3 Flash 0.596, Gemma 4 26B A4B 0.813, Qwen 3.5 Plus 0.958, DeepSeek V4 Flash 1.025, Gemma 4 26B A4B (без контекста) 1.265, DeepSeek V4 Flash (без контекста) 1.647, Qwen 3.5 Flash 2.198, DeepL 4.963, DeepL (без контекста) 5.717, Google Translation Basic 5.998.](docs/images/performance/1.png)

- Эксперимент проводился с использованием фреймворка Microsoft Gemba MQM.
- Для максимального приближения к реальному разговору использовалась многоходовая среда.
- Полные результаты — [здесь](https://github.com/kapitalismho/korean-llm-context-translation-benchmark).

## Стоимость

### Количество использований за доллар

#### Рекомендуемые модели

| LLM \ ASR | Qwen ASR (локальный) | Qwen ASR (облачный) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | 14 380 | 2 920 | 3 710 | 1 180 |
| **DeepSeek V4 Flash** | 19 410 | 3 080 | 3 980 | 1 210 |

#### Другие модели

| LLM \ ASR | Qwen ASR (локальный) | Qwen ASR (облачный) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 31B (Cerebras)** | 920 | 730 | 770 | 540 |
| **DeepSeek V4 Pro** | 6 400 | 2 330 | 2 810 | 1 070 |
| **Gemini 3 Flash** | 1 710 | 1 170 | 1 280 | 740 |
| **Gemini 3.1 Flash-Lite** | 3 430 | 1 770 | 2 030 | 940 |
| **Qwen 3.5 Plus** | 7 460 | 2 460 | — | — |
| **Локальные LLM** | Без ограничений | 3 660 | 5 000 | 1 290 |

### Стоимость одного высказывания

#### Рекомендуемые модели

| LLM \ ASR | Qwen ASR (локальный) | Qwen ASR (облачный) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | ~$0,00007 | ~$0,0003 | ~$0,0003 | ~$0,0008 |
| **DeepSeek V4 Flash** | ~$0,00005 | ~$0,0003 | ~$0,0003 | ~$0,0008 |

#### Другие модели

| LLM \ ASR | Qwen ASR (локальный) | Qwen ASR (облачный) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 31B (Cerebras)** | ~$0,0011 | ~$0,0014 | ~$0,0013 | ~$0,0019 |
| **DeepSeek V4 Pro** | ~$0,0002 | ~$0,0004 | ~$0,0004 | ~$0,0009 |
| **Gemini 3 Flash** | ~$0,0006 | ~$0,0009 | ~$0,0008 | ~$0,0014 |
| **Gemini 3.1 Flash-Lite** | ~$0,0003 | ~$0,0006 | ~$0,0005 | ~$0,0011 |
| **Qwen 3.5 Plus** | ~$0,0001 | ~$0,0004 | — | — |
| **Локальные LLM** | $0 | ~$0,0003 | ~$0,0002 | ~$0,0008 |

*   *Расчёт: (900 входных токенов + 12 выходных) × 1,2 среднее число вызовов LLM на одно высказывание.*
*   *Количество использований за доллар рассчитано по неокруглённым значениям.*
*   *Все стоимости приблизительны.*
*   *DeepSeek предполагает 70% попаданий в кэш.*
*   *Стоимость API Qwen — по региону Пекин.*
*   *Цены актуальны на 25 мая 2026 г. / Режим быстрого ответа.*

### Бесплатные кредиты

| Сервис | Бесплатный кредит | Срок | Примечание |
|--------|------------|------|------|
| **Deepgram** | $200 | Без ограничений | — |
| **Google AI Studio** | $10 | 1 год | Ежемесячно для подписчиков Gemini |
| **Alibaba Cloud** | 1 млн токенов на модель | 90 дней | Регион Сингапур |
| **Alibaba Cloud** | ¥300 | 1 год | Студенты в Китае |
| **Cerebras** | 1 млн токенов ежедневно | Без ограничений | Лимит 5 вызовов в минуту |

---

# Если у вас возникли вопросы или что-то непонятно — пишите в [Twitter/X](https://x.com/kapitalismho).

## Использование

1. Скачайте последнюю версию со [страницы загрузки](https://github.com/kapitalismho/PuriPuly-heart/releases/latest).
2. Установите PuriPuly.
3. Нажмите кнопку **STT**.
4. Нажмите кнопку **TRANS** и авторизуйтесь через Discord.
5. Нажмите кнопку **Subtitles** для включения субтитров в VR.
6. *(Необязательно)* Нажмите кнопку **Peer** для включения перевода речи собеседника.

   > Для перевода речи собеседника требуется тихое окружение. В VRChat используйте Earmuff.

7. Включите OSC в VRChat: Меню действий → Настройки → OSC → Включить.

### Если захват звука не работает

Откройте **Настройки > Основные** и выполните следующее:

1. Измените **Audio Host API** на **Auto** или **MME**.
2. Выберите правильный микрофон.
3. Перезапустите приложение.

---

### Примечание для пользователей из Китая

Если Soniox / Gemini / Deepgram заблокированы в вашем регионе, используйте:

- STT: **Qwen ASR**
- LLM: **DeepSeek V4 Flash**

   > Авторизация доступна через QQ вместо Discord.

---

### Использование собственных API-ключей

Следуйте инструкции для нужного сервиса.

Для переводческой LLM рекомендуется модель Gemma 4 через OpenRouter.

Кстати, пока настраиваете — почему бы не настроить и ASR?
PuriPuly показывает лучший результат в паре с облачным распознаванием речи.
Даже с одним и тем же Qwen ASR разница между локальным и облачным распознаванием заметна.

Рекомендуем начать с Deepgram.
Регистрация даёт $200 бесплатных кредитов.

<details>
<summary><h3>OpenRouter</h3></summary>

1. Установите параметры, обведённые красным, как на скриншоте.
   ![step0](docs/images/openrouter/0.png)

2. В приложении нажмите кнопку, обведённую красным.
   ![step1](docs/images/openrouter/1.png)

3. Войдите в OpenRouter.
   ![step2](docs/images/openrouter/2.png)

4. Нажмите обведённую кнопку, чтобы выйти из экрана оплаты.
   ![step3](docs/images/openrouter/3.png)

5. Нажмите **Authorize**.
   ![step4](docs/images/openrouter/4.png)

6. Пополните баланс на нужную сумму.
   ![step5](docs/images/openrouter/5.png)

<details>
<summary><h3>Если кнопка Authorize не сработала</h3></summary>

Попробуйте ещё раз или создайте API-ключ вручную:

6. Нажмите на аккаунт в правом верхнем углу → вкладка API Keys → кнопка Create.
   ![step6](docs/images/openrouter/6.png)

7. Нажмите Create.
   ![step7](docs/images/openrouter/7.png)

8. Скопируйте API-ключ и вставьте его на вкладку API переводчика.
   ![step8](docs/images/openrouter/8.png)

</details>

</details>

<details>
<summary><h3>DeepSeek</h3></summary>

1. Установите параметры, обведённые красным, как на скриншоте.
   ![step0](docs/images/deepseek/0.png)

2. Перейдите на [официальный сайт DeepSeek](https://www.deepseek.com/en/) и нажмите **Access API**.
   ![step1](docs/images/deepseek/1.png)

3. Войдите на сайт.
   ![step2](docs/images/deepseek/2.png)

4. Перейдите на вкладку API Keys → **Create new API Keys**.
   ![step3](docs/images/deepseek/3.png)

5. Скопируйте API-ключ и вставьте его на вкладку API переводчика.
   ![step4](docs/images/deepseek/4.png)

6. Перейдите на вкладку Top Up и пополните баланс.
   ![step5](docs/images/deepseek/5.png)

</details>

<details>
<summary><h3>Deepgram</h3></summary>

1. Войдите в [Deepgram Console](https://console.deepgram.com/).
   ![step1](docs/images/deepgram/1.png)

2. Если видите приветствие — нажмите **Skip**.
   ![step2](docs/images/deepgram/2.png)

3. Выберите **STT (Speech-to-Text)**.
   ![step3](docs/images/deepgram/3.png)

4. В меню API Keys нажмите **Create a New API Key**.
   ![step4](docs/images/deepgram/4.png)

5. Введите имя ключа (например, `puripuly`) и создайте.
   ![step5](docs/images/deepgram/5.png)

6. Скопируйте ключ и вставьте в настройки PuriPuly.
   ![step6](docs/images/deepgram/6.png)

</details>

<details>
<summary><h3>Gemini</h3></summary>

1. Перейдите в [Google AI Studio](https://aistudio.google.com/apikey) → **Get API key**.
   ![step1](docs/images/gemini/1.png)

2. Создайте новый проект.
   ![step2](docs/images/gemini/2.png)

3. Введите любое имя.
   ![step3](docs/images/gemini/3.png)

4. Выберите проект → **Create key**.
   ![step4](docs/images/gemini/4.png)

5. Нажмите на обведённую область.
   ![step5](docs/images/gemini/5.png)

6. Скопируйте ключ.
   ![step6](docs/images/gemini/6.png)

7. *(Рекомендуется)* Нажмите жёлтую кнопку **Set Up Billing** для перехода на платный тариф.
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
   - [Материковый Китай](https://bailian.console.aliyun.com/cn-beijing)
   - [Остальной мир](https://bailian.console.alibabacloud.com)

2. Войдите. Убедитесь, что выбран правильный регион (например, Пекин).
   ![step2](docs/images/qwen/1.png)

3. Нажмите **значок шестерёнки** в правом верхнем углу.
   ![step3](docs/images/qwen/2.png)

4. Создайте рабочее пространство → страница **API-KEY**.
   ![step4](docs/images/qwen/3.png)

5. **Create API Key**.
   ![step5](docs/images/qwen/4.png)

6. Назначьте аккаунт и рабочее пространство → OK.
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

3. Нажмите **Add Funds** для привязки оплаты.
   ![step3](docs/images/soniox/3.png)

4. После пополнения перейдите в **API Keys**.
   ![step4](docs/images/soniox/4.png)

5. Создайте новый API Key.
   ![step5](docs/images/soniox/5.png)

6. Скопируйте ключ и вставьте в настройки PuriPuly.
   ![step6](docs/images/soniox/6.png)

</details>

<details>
<summary><h3>Cerebras</h3></summary>

1. Перейдите на [Cerebras](https://www.cerebras.ai/) → **Get started**.
   ![step1](docs/images/cerebras/1.png)

2. Войдите.
   ![step2](docs/images/cerebras/2.png)

3. Выберите тариф. Рекомендуем начать с бесплатного.
   ![step3](docs/images/cerebras/3.png)

4. Скопируйте API-ключ и вставьте в PuriPuly.
   ![step4](docs/images/cerebras/4.png)

<details>
<summary><h3>Для перехода на платный тариф</h3></summary>

5. Перейдите на вкладку **Billing**.
   ![step5](docs/images/cerebras/5.png)

6. Введите имя.
   ![step6](docs/images/cerebras/6.png)

7. Пополните баланс.
   ![step7](docs/images/cerebras/7.png)

</details>

</details>

---

## Разработка

### Среда разработки

| Область | Рекомендуемое окружение |
|---|---|
| Python-приложение | Windows |
| VR-оверлей | Windows |
| Сервис-брокер | Linux / WSL |

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
# Показывает скрытые элементы интерфейса
python -m puripuly_heart.main run-gui --debug-ui-preview
```

### Тесты и линтинг

```bash
black src tests          # Форматирование
ruff check src tests     # Линтинг
python -m pytest         # Тесты (рекомендуется внутри venv)
```

### VR-оверлей

VR-оверлей субтитров собирается из Rust-проекта в `native/overlay/`.

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

### Сервис-брокер

Подробности — в `broker/README.md`.

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

## Особая благодарность

SUI\_32C, Nagikokoro, motoka96, \_Ykol魚, kascr\_, Just Monika V, FLUVIA, Han โชเล่ย์, EA\_PE, Ephedrine, ~ eri ~

---

## Лицензия

[AGPL-3.0-or-later](LICENSE)

Сторонние лицензии и уведомления: `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`
