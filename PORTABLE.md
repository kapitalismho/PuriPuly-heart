# PuriPuly-heart Portable

Портативная версия [PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart) — работает с USB-флешки без установки.

> Fork с веткой `portable` — содержит минимальные изменения для поддержки portable-режима.

## Быстрый старт

1. Скачайте ZIP из [Releases](../../releases)
2. Распакуйте в любую папку
3. Запустите `start.exe`

## Структура

```
PuriPulyHeart-Portable/
├── start.exe           ← тихий запуск
├── start-debug.exe     ← запуск с консолью (для отладки)
├── python/             ← embedded Python 3.12 + зависимости
├── data/               ← все данные приложения
│   ├── settings.json
│   ├── secrets.json
│   ├── models/         ← ASR модель (~987 MB)
│   └── ...
└── app/                ← исходники PuriPuly-heart
```

## Изменения в коде (минимальные)

| Файл | Что |
|------|-----|
| `config/paths.py` | Проверка `PURIPULY_HEART_DATA_DIR` в `user_config_dir()` |
| `app/wiring.py` | Автопереключение на EncryptedFile для секретов |
| `config/settings.py` | Дефолтный backend = ENCRYPTED_FILE в portable-режиме |

## Как работает

`start.exe` выставляет `PURIPULY_HEART_DATA_DIR` и запускает embedded Python. Приложение видит env var и перенаправляет все пути в `data/`. Без env var — работает как обычно (установленная версия).

## Сборка из исходников

```bat
:: Скачать embedded Python
curl -L -o python.zip https://www.python.org/ftp/python/3.12.7/python-3.12.7-embed-amd64.zip
powershell "Expand-Archive python.zip -DestinationPath python"
del python.zip

:: Настроить (раскомментировать import site в python312._pth, добавить ..\app\src)
copy python312._pth python\

:: Установить pip и зависимости
curl -L -o python\get-pip.py https://bootstrap.pypa.io/get-pip.py
python\python.exe python\get-pip.py
python\python.exe -m pip install -r requirements.txt

:: Скомпилировать лаунчер
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /target:winexe /out:start.exe start.cs
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /target:exe /out:start-debug.exe start.cs
```

---

Based on [kapitalismho/PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart)

*Created by: mimo-pro*
