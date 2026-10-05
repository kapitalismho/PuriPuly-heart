<p align="center">
  <img src="../../src/puripuly_heart/data/icons/icon.png" alt="PuriPuly — двусторонний голосовой переводчик для VRChat в реальном времени" width="128" />
</p>

<h1 align="center">PuriPuly<br>
  <sub>двусторонний голосовой переводчик для VRChat в реальном времени</sub>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.8.0-blue" alt="Version" />
  <img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue" alt="License: AGPL-3.0-or-later" />
  <img src="https://img.shields.io/badge/python-3.12-yellow" alt="Python" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey" alt="Platform" />
</p>

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

## Что такое PuriPuly?

PuriPuly — двусторонний голосовой переводчик для Windows: переводит в реальном времени вашу речь и речь собеседника.
Мы стремимся к естественному переводу с помощью LLM.
За рамками сухого перевода — к настоящему общению между людьми.
Работает во многих средах, включая VRChat и Discord.

- **Перевод на основе LLM** — сленг, разговорная речь, твои и ваши — всё звучит естественно.
- **Память контекста** — перевод помнит, о чём вы говорили раньше, и не теряет нить беседы.
- **Голосовой перевод в обе стороны** — переводит и вашу речь, и речь собеседника. Есть субтитры в VR.
- **Запуск через Discord** — просто установите и начните пользоваться, без сложной настройки.
- **Самый мощный локальный стек** — от Parakeet до Gemma 4 E4B: только самые эффективные модели на сегодня.

## Вопросы и ответы

- **Насколько хорошее качество перевода?**
→ Можно без труда вести самые глубокие разговоры, на какие только способны люди. К тому же он с большим отрывом опережает традиционные коммерческие сервисы перевода. Подробности — в разделе «Сравнение перевода» ниже.

- **Сколько времени от фразы до перевода?**
→ В оптимальных условиях задержка составляет около 1 секунды с момента, когда собеседник закончил говорить.

- **Это стоит денег?**
→ Да, но не сразу. Новым пользователям даётся бесплатный кредит, а потом цены копеечные — тысячи переводов за $1. А при использовании локальных моделей всё можно использовать бесплатно.

- **Нужен ли API-ключ?**
→ Да, но не сразу. Просто установите и авторизуйтесь через Discord, чтобы начать пользоваться.

- **Распознавание речи работает медленно**
→ При использовании локального ASR нехватка вычислительных ресурсов может замедлить обработку. В таком случае рекомендуем переключиться на облачный сервис STT.

- **Как обрабатываются личные данные?**
→ Голос и текст разговоров не отправляются на серверы Puripuly. Кроме того, весь исходный код открыт в этом репозитории, поэтому вы можете напрямую проверить сетевую активность программы.

### [📥 Скачать](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)

---

## Сравнение перевода

![Средний штраф за предложение для полного конвейера распознавания речи и перевода. С корейского на английский / японский / китайский (упрощённый), 216 многоходовых примеров, оценка Gemba MQM; чем ниже, тем лучше. Синие столбцы — сочетания, доступные в PuriPuly: Gemini Transcribe → Luna (0.676), Soniox STT → Luna (0.942), Qwen ASR 1.7B → Gemma 26B (1.024), Gemini Transcribe → Gemma 26B (1.084), Soniox STT → Gemma 26B (1.293), Qwen ASR 0.6B → Gemma 26B (2.374). Оранжевые столбцы — внешние решения для сравнения: Qwen 3.8 Live Translate (2.108), Gemini 3.5 Live Translate (3.754), Soniox Translate (4.989). Модель-оценщик: Gemini 3.7 Flash.](docs/images/performance/1.png)

![Средний штраф за предложение. С корейского на английский / японский / китайский (упрощённый), 216 многоходовых примеров, оценка Gemba MQM; чем ниже, тем лучше. Синие столбцы — модели, доступные в PuriPuly: GPT 6 Luna (0.130), Gemma 4 26B A4B (0.387), DeepSeek-V4 Flash 0731 (0.571), Gemma 4 E4B QAT Q4 (1.577). Оранжевые столбцы — внешние решения для сравнения: Qwen 3.8 Live Translate (1.392), Papago (2.699), Gemini 3.5 Translate (2.991), Soniox Translate (3.473), DeepL (3.914), Google Translation (5.731). Для Qwen 3.8, Gemini 3.5 и Soniox отобраны только результаты с долей ошибочных символов (CER) не выше 5%. Модель-оценщик: Gemini 3.7 Flash.](docs/images/performance/2.png)

- Синие столбцы — модели, доступные в PuriPuly.
- Для эксперимента использован фреймворк Microsoft Gemba MQM.
- Тесты шли в диалоговом формате — ближе к реальному разговору.
- Полные результаты — [здесь](https://github.com/kapitalismho/korean-llm-context-translation-benchmark).

## Стоимость

### Переводов за доллар

#### Рекомендуемые модели

| LLM \ ASR | Локальный ASR | Клауд фри тир | Soniox | Qwen Audio |
|---|---|---|---|---|
| **Gemma 4 E4B (локальный)** | Без ограничений | Без ограничений | 5 000 | 7 260 |
| **Gemma 4 26B A4B + 31B** | 13 940 | 13 940 | 3 680 | 4 770 |
| **DeepSeek V4.1 Flash** | 16 800 | 16 800 | 3 860 | 5 070 |
| **GPT 6 Luna** | 8 680 | 8 680 | 3 170 | 3 950 |

#### Другие модели

| LLM \ ASR | Локальный ASR | Клауд фри тир | Soniox | Qwen Audio |
|---|---|---|---|---|
| **DeepSeek V4 Flash (OpenRouter)** | 17 020 | 17 020 | 3 860 | 5 090 |
| **Gemini 3.8 Flash** | 1 160 | 1 160 | 940 | 1 000 |
| **Qwen 3.8 Flash** | 7 460 | 7 460 | 2 990 | 3 680 |

### Цена одной фразы

#### Рекомендуемые модели

| LLM \ ASR | Локальный ASR | Клауд фри тир | Soniox | Qwen Audio |
|---|---|---|---|---|
| **Gemma 4 E4B (локальный)** | $0 | $0 | ~$0,0002 | ~$0,00014 |
| **Gemma 4 26B A4B + 31B** | ~$0,00007 | ~$0,00007 | ~$0,0003 | ~$0,00021 |
| **DeepSeek V4.1 Flash** | ~$0,00006 | ~$0,00006 | ~$0,0003 | ~$0,00020 |
| **GPT 6 Luna** | ~$0,00012 | ~$0,00012 | ~$0,0003 | ~$0,00025 |

#### Другие модели

| LLM \ ASR | Локальный ASR | Клауд фри тир | Soniox | Qwen Audio |
|---|---|---|---|---|
| **DeepSeek V4 Flash (OpenRouter)** | ~$0,00006 | ~$0,00006 | ~$0,0003 | ~$0,00020 |
| **Gemini 3.8 Flash** | ~$0,0009 | ~$0,0009 | ~$0,0011 | ~$0,0010 |
| **Qwen 3.8 Flash** | ~$0,0001 | ~$0,0001 | ~$0,0003 | ~$0,00027 |

*   *Расчёт: (900 входных + 12 выходных токенов) × 1,2 вызова LLM на фразу.*
*   *Переводов за доллар — по неокруглённым значениям.*
*   *Все цены приблизительны.*
*   *DeepSeek V4.1 Flash — с учётом 70% попаданий в кэш, V4 Flash (OpenRouter) — 60%.*
*   *GPT 6 Luna — по тарифам OpenAI API, без скидки за кэш. При подключении через ChatGPT вместо оплаты API расходуются лимиты Codex.*
*   *Qwen — по тарифам региона Пекин.*
*   *Цены на 25 сентября 2026 г.*

### Бесплатные кредиты

| Сервис | Бесплатный кредит | Срок | Примечание |
|--------|------------|------|------|
| **Deepgram** | $200 | Без ограничений | Карта не нужна |
| **ElevenLabs** | 10 000 кредитов | Сброс каждый месяц | Карта не нужна |
| **Gemini 3.5 Transcribe** | Бесплатный тариф | Без ограничений | На бесплатном тарифе фактически без ограничений |
| **Alibaba Cloud** | 1 млн токенов на модель | 90 дней | Регион Сингапур |
| **Alibaba Cloud** | ¥300 | 1 год | Студенты в Китае |

---

## Локальные модели

В PuriPuly встроены следующие локальные модели. Также можно подключать API, совместимые с OpenAI.
GPU-инференс работает на Vulkan. Подходит для любого производителя — хоть Radeon, хоть Arc.

**ASR**

| Модель | Среда выполнения | Квантование |
|---|---|---|
| Parakeet TDT 0.6B v3 | CPU | INT8 |
| Parakeet TDT-CTC 0.6B (ja) | CPU | INT8 |
| Qwen3-ASR 0.6B | CPU | INT8 |
| Qwen3-ASR 1.7B | GPU | Q6_K |
| OpenAI-совместимый API | — | — |

**LLM**

| Модель | Среда выполнения | Квантование |
|---|---|---|
| Gemma 4 E4B IT QAT | CPU / GPU | UD Q4_K_XL |
| OpenAI-совместимый API | — | — |

---

# Если возникнут проблемы, напишите в [Twitter/X](https://x.com/kapitalismho).

## Использование

1. Скачайте последнюю версию со [страницы загрузки](https://github.com/kapitalismho/PuriPuly-heart/releases/latest).
2. Установите PuriPuly.
3. Нажмите кнопку **TALK**.
4. Нажмите кнопку **TRANS** и авторизуйтесь через Discord.
5. Нажмите кнопку **CAPTIONS** для включения субтитров в VR.
6. *(Необязательно)* Нажмите кнопку **LISTEN** для включения перевода речи собеседника.

   > Для перевода речи собеседника требуется тихое окружение. В VRChat используйте Earmuff.

7. Включите OSC в VRChat: Меню действий → Настройки → OSC → Включить.

### Если захват звука не работает

Откройте **Настройки > Основные** и выполните следующее:

1. Измените **Audio Host API** на **Auto** или **MME**.
2. Выберите правильный микрофон.
3. Перезапустите приложение.

---

### Пользователям из Китая

Если Soniox / Gemini / Deepgram у вас заблокированы, попробуйте такую связку:

- STT: **Qwen Audio**
- LLM: **DeepSeek V4.1 Flash**

   > Вместо Discord можно авторизоваться через QQ.

---

### Свои API-ключи

Выберите нужный сервис и следуйте инструкции.

Для перевода рекомендуем Gemma 4 через OpenRouter.

А заодно настройте и распознавание речи!
PuriPuly работает лучше всего с облачным STT.
Даже один и тот же Qwen ASR на локале и в облаке заметно отличается по качеству.

Начните с Deepgram — при регистрации дают $200.

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
   Переход на платный тариф может занять некоторое время.
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

4. Soniox требует предоплаты. После пополнения перейдите в **API Keys**.
   ![step4](docs/images/soniox/4.png)

5. Создайте новый API Key.
   ![step5](docs/images/soniox/5.png)

6. Скопируйте ключ и вставьте в настройки PuriPuly.
   ![step6](docs/images/soniox/6.png)

</details>


---

## Архитектура

См. [`docs/architecture.md`](docs/architecture.md).

## Дорожная карта

Предстоящая работа отслеживается публично на [доске проекта PuriPuly](https://github.com/users/kapitalismho/projects/2).

---

## Разработка

Нужны Windows x64, обычный CPython 3.14 с GIL и [uv](https://docs.astral.sh/uv/). Выполняйте команды из корня репозитория.

### Установка

```powershell
uv sync --frozen --extra dev
```

### GUI

```powershell
uv run python -m puripuly_heart.main run-gui
```

### CLI

CLI позволяет запускать приложение без GUI и управлять уже работающим приложением. Команды описаны в [руководстве по CLI](docs/cli.md).

```powershell
uv run python -m puripuly_heart.main cli --help
```

### Проверка

```powershell
uv run black --check src tests
uv run ruff check src tests
uv run python -m pytest
```

[Разработка брокера (Linux)](broker/README.md) · [Разработка VR-оверлея (Windows)](native/overlay/README.md)

---

## Разработчик

[salee](https://github.com/kapitalismho)

---

## Участники

[RICHARDwuxiaofei](https://github.com/RICHARDwuxiaofei)
[fzcfweasdferttgg-png](https://github.com/fzcfweasdferttgg-png)

---

## Special Thanks

SUI\_32C, Nagikokoro, motoka96, \_Ykol魚, kascr\_, Just Monika V, FLUVIA, Han โชเล่ย์, EA\_PE, Ephedrine, ~ eri ~, fzcfweasdferttgg-png, Welcius, nunu299, 梅雨Shiro

---

## Политики

- [Code signing policy](CODE_SIGNING.md)
- [Политика конфиденциальности](PRIVACY.md)

---

## Лицензия

[AGPL-3.0-or-later](LICENSE)

Сторонние лицензии и уведомления: `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`
