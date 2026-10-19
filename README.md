# ⚡ Parakeet Redux Pro: Local Transcriber, Subtitles & AI Editor

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Moondream AI](https://img.shields.io/badge/Moondream-Parakeet--Redux-7C3AED.svg?style=flat)](https://huggingface.co/moondream/parakeet-redux)
[![Google Gemini](https://img.shields.io/badge/Gemini_API-3.8_Flash-4285F4.svg?style=flat&logo=google)](https://aistudio.google.com/)
[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-10B981.svg?style=flat)](https://creativecommons.org/licenses/by/4.0/)
[![GitHub Topics](https://img.shields.io/badge/Topics-Speech--to--Text%20%7C%20Subtitles%20%7C%20Reels%20%7C%20Shorts%20%7C%20Gemini-blue)](https://github.com/alexsandrk20-cell/Parakeet_Redux_Pro/topics)

Сверхбыстрое локальное распознавание речи на базе **`moondream/parakeet-redux`** (1.58-bit Ternary AI) с поддержкой **GPU CUDA**, **пословных таймкодов**, **караоке-субтитров для видеомонтажа** и умной постобработкой через **Google Gemini AI**.

> [!NOTE]
> **Благодарность автору исходного проекта:**
> Проект основан на оригинальном репозитории [47thtechcorner/RayCodes_Parakeet_Redux](https://github.com/47thtechcorner/RayCodes_Parakeet_Redux) от [RayCodes](https://youtu.com/nwx4ph8iELA).

---

## 🌟 Новые возможности и улучшения

1. ⏱️ **Пословные временные метки (Word-Level Timestamps):**
   * Получение точного времени начала и окончания каждого слова с точностью до миллисекунд.
   * Интерактивная таблица в веб-интерфейсе с поиском и сортировкой по всем словам записи.

2. 🎬 **Экспорт субтитров для видеомонтажа:**
   * **Стандартные субтитры (`.srt`, `.vtt`)** — по предложениям для DaVinci Resolve, Adobe Premiere Pro, Final Cut.
   * **Пословные караоке-субтитры (`_words.srt`)** — каждое слово отдельным блоком с точным таймингом (для трендовой анимации слов в Reels, Shorts, TikTok и CapCut).
   * **Полный JSON (`.json`)** и структурированный отчет (`.md`).

3. ✂️ **Умная очистка от знаков препинания (с сохранением дефисов):**
   * Удаление точек, запятых, тире, кавычек, скобок, восклицательных и вопросительных знаков для чистого видеоряда.
   * **Сохранение внутрисловных дефисов:** сложные слова вроде *«наконец-то»*, *«из-за»*, *«кто-то»*, *«по-русски»* сохраняют дефис и не слипаются в одно слово.

4. 🔠 **Управление регистром (UPPERCASE):**
   * Перевод всех слов в **ВЕРХНИЙ РЕГИСТР (ЗАГЛАВНЫЕ)**, строчные или с заглавной буквы.

5. 🚫 **Удаление слов-паразитов (Filler Words Filter):**
   * Встроенный словарь русских и английских звуков заминки (*ээ*, *эм*, *мм*) и слов-паразитов (*ну*, *короче*, *типа*, *как бы*, *в общем*, *так сказать* и др.).
   * Возможность редактирования списка слов-паразитов прямо в веб-интерфейсе.

6. 🤖 **Интеллектуальная AI-правка (Google Gemini и OpenAI-совместимый API):**
   * Поддержка **Google Gemini** (`gemini-3.8-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`) и **любого OpenAI-совместимого API** (OpenAI, OpenRouter, DeepSeek, vLLM, локальные LLM).
   * Автоматическое исправление смысловых и фонетических ошибок ASR, профессиональных терминов.
   * Контекстное удаление оговорок и заминок с точным сохранением временных меток слов.
   * Встроенная проверка подключения к API с измерением задержки в один клик.
   * Вкладка интерактивного сравнения «До и После».
   * Ключи и настройки сохраняются локально в `.env` и никогда не попадают в git.

7. ⚡ **Аппаратное ускорение на GPU (NVIDIA CUDA):**
   * Оптимизированный инференс (транскрибация 30 секунд аудио занимает всего ~1.4 сек на RTX 3060).

8. 📱 **Клиповые субтитры 9:16 (Shorts / Reels / TikTok) с режимом «Без пауз» (Gapless):**
   * **Умная компоновка строки:** короткие слова объединяются до 2–4 слов (до 20 знаков), идеально под вертикальный экран смартфона в 1 строку на блок.
   * **Длинные слова отдельно:** слова длиннее 12 символов (*«ПРИВЕТСТВОВАТЬ»*) автоматически ставятся отдельным блоком.
   * **Правило предлогов и союзов:** предлоги (*«с людьми»*, *«про глоссарий»*, *«на уроке»*, *«по автоматическому»*) всегда начинают новую строку и никогда не висят в конце фразы.
   * **Защита от одиноких местоимений:** слова вроде *«я»*, *«мы»*, *«он»* никогда не изолируются в одну строку, а связываются с фразой.
   * **Замена «ё» на «е»:** автоматическая замена по всему тексту и субтитрам.
   * **⚡ Выделенная кнопка «Без пауз» (Gapless SRT):** таймкоды без разрывов (конец каждого титра равен началу следующего), субтитры не моргают и не оставляют пустой экран в Premiere Pro, CapCut и DaVinci Resolve.

9. 📱 **Интерактивный визуальный предпросмотр 9:16 в окне смартфона:**
   * Реалистичный мокап экрана телефона с безопасной зоной соцсетей (Reels, Shorts, TikTok).
   * Плеер таймлайна с управлением воспроизведением, перемоткой и скраббером.
   * Переключение трендовых стилей субтитров на лету: 🟡 Hormozi Gold, ⚪ Clean White, 🟢 Neon Lime, 🔵 Cyber Cyan.
   * Настройка размера шрифта (S/M/L) и синхронизированный интерактивный список с поиском.

10. 🎯 **Библиотека системных промтов и редактор пресетов:**
    * Встроенные профили: *Универсальный*, *Клиповый 9:16*, *Подкасты*, *IT и разработка*, *Пользовательский*.
    * Возможность редактирования, сохранения и сброса промтов прямо из интерфейса.

11. 🚀 **Удобный запуск в один клик:**
    * Скрипт `run.bat` для быстрого запуска под Windows.

---

## 🛠️ Системные требования

* **ОС:** Windows 10/11, Linux или macOS.
* **Python:** 3.10 – 3.12 (рекомендуется Python 3.12).
* **FFmpeg:** Установлен в системе и доступен в PATH.
  * Установка на Windows через winget:
    ```powershell
    winget install --id FFmpeg-FFmpeg.FFmpeg -e
    ```
* **Видеокарта (опционально):** NVIDIA GPU с поддержкой CUDA для максимальной скорости (также поддерживается запуск на CPU).

---

## 🚀 Установка и настройка

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/alexsandrk20-cell/Parakeet_Redux_Pro.git
   cd Parakeet_Redux_Pro
   ```

2. **Создайте виртуальное окружение:**
   ```bash
   # С использованием стандартного venv:
   python -m venv .venv

   # Активация на Windows:
   .venv\Scripts\activate

   # Активация на Linux/macOS:
   source .venv/bin/activate
   ```

3. **Установите зависимости:**
   * Для систем с **NVIDIA GPU (CUDA 12.4+)**:
     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cu124
     pip install -r requirements.txt
     ```
   * Для запуска только на **CPU**:
     ```bash
     pip install -r requirements.txt
     ```

4. **Настройка AI-провайдера (опционально):**
   * Скопируйте `.env.example` в `.env`:
     ```bash
     copy .env.example .env
     ```
   * Укажите ключи в `.env`:
     * Для **Google Gemini**: бесплатный ключ из [Google AI Studio](https://aistudio.google.com/app/apikey):
       ```env
       AI_PROVIDER=gemini
       GEMINI_API_KEY=ваш_ключ_здесь
       ```
     * Для **OpenAI-совместимого API**:
       ```env
       AI_PROVIDER=openai_compatible
       OPENAI_BASE_URL=https://api.openai.com/v1
       OPENAI_API_KEY=ваш_ключ_здесь
       OPENAI_MODEL=gpt-4o-mini
       ```
   *(Также провайдер и ключи можно настраивать и проверять прямо в боковом меню приложения).*

---

## 💻 Запуск приложения

* **Windows:** Просто дважды кликните по файлу **`run.bat`** (или создайте ярлык на рабочем столе).
* **Через терминал:**
  ```powershell
  streamlit run app.py
  ```

Интерфейс откроется в браузере по адресу: `http://localhost:8501`.

---

## 📁 Структура проекта

```
├── app.py              # Основное приложение Streamlit с полной логикой ASR и Gemini
├── run.bat             # Скрипт быстрого запуска под Windows
├── requirements.txt    # Зависимости проекта
├── .env.example        # Пример конфигурационного файла для API-ключа
├── .gitignore          # Исключение секретов, виртуальных окружений и медиа
└── README.md           # Документация проекта
```

---

## 📄 Лицензия и атрибуция

Проект распространяется под лицензией [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).  
Оригинальная разработка: [47thtechcorner/RayCodes_Parakeet_Redux](https://github.com/47thtechcorner/RayCodes_Parakeet_Redux).  
Модель ASR: [Moondream Parakeet-Redux](https://huggingface.co/moondream/parakeet-redux).

---

## 🏷️ Теги и ключевые слова для поиска (Topics & Keywords)

`speech-to-text` • `transcription` • `subtitles` • `subtitles-generator` • `srt` • `vtt` • `shorts` • `reels` • `tiktok` • `parakeet` • `moondream` • `gemini` • `google-gemini` • `cuda` • `karaoke` • `capcut` • `davinci-resolve` • `premiere-pro` • `video-editing` • `streamlit` • `audio-processing` • `asr` • `ffmpeg` • `word-level-timestamps` • `gapless-subtitles` • `voice-recognition` • `subtitles-editor`

