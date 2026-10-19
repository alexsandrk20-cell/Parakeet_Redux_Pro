# ⚡ Parakeet Redux Pro: Local Transcriber, Subtitles & AI Editor

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Moondream AI](https://img.shields.io/badge/Moondream-Parakeet--Redux-7C3AED.svg?style=flat)](https://huggingface.co/moondream/parakeet-redux)
[![Google Gemini](https://img.shields.io/badge/Gemini_API-3.8_Flash-4285F4.svg?style=flat&logo=google)](https://aistudio.google.com/)
[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-10B981.svg?style=flat)](https://creativecommons.org/licenses/by/4.0/)

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

3. ✂️ **Очистка от знаков препинания:**
   * Опция удаления точек, запятых, тире и кавычек для создания чистого динамического видеоряда.

4. 🔠 **Управление регистром (UPPERCASE):**
   * Перевод всех слов в **ВЕРХНИЙ РЕГИСТР (ЗАГЛАВНЫЕ)**, строчные или с заглавной буквы.

5. 🚫 **Удаление слов-паразитов (Filler Words Filter):**
   * Встроенный словарь русских и английских звуков заминки (*ээ*, *эм*, *мм*) и слов-паразитов (*ну*, *короче*, *типа*, *как бы*, *в общем*, *так сказать* и др.).
   * Возможность редактирования списка слов-паразитов прямо в веб-интерфейсе.

6. 🤖 **Интеграция с Google Gemini AI (`gemini-3.8-flash`):**
   * Автоматическое исправление смысловых и фонетических ошибок ASR, профессиональных терминов.
   * Контекстное удаление оговорок и заминок с сохранением временных меток.
   * Вкладка интерактивного сравнения «До и После».
   * Ключ API сохраняется локально в файл `.env` и никогда не утекает в репозиторий.

7. ⚡ **Аппаратное ускорение на GPU (NVIDIA CUDA):**
   * Оптимизированный инференс (транскрибация 30 секунд аудио занимает всего ~1.4 сек на RTX 3060).

8. 🚀 **Удобный запуск в один клик:**
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

4. **Настройка Gemini API (опционально):**
   * Скопируйте `.env.example` в `.env`:
     ```bash
     copy .env.example .env
     ```
   * Вставьте ваш бесплатный ключ из [Google AI Studio](https://aistudio.google.com/app/apikey):
     ```env
     GEMINI_API_KEY=ваш_ключ_здесь
     ```
   *(Также ключ можно ввести прямо в боковом меню приложения).*

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
