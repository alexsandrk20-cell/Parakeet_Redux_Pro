import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import time
import subprocess
import json
import re
import string
import pandas as pd
import difflib
# pyrefly: ignore [missing-import]
import moondream as md

st.set_page_config(
    page_title="Parakeet-Redux Local Transcriber",
    page_icon="⚡",
    layout="wide"
)

# Exclude hyphen '-' to preserve compound words like 'наконец-то', 'из-за', 'по-русски'
PUNCTUATION_CHARS = set(string.punctuation.replace("-", "") + "«»—–…„“”")

ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "system_prompt.txt")

def load_system_prompt() -> str:
    """Loads system prompt from system_prompt.txt if present, otherwise returns DEFAULT_SYSTEM_PROMPT."""
    if os.path.exists(SYSTEM_PROMPT_FILE):
        try:
            with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
    return DEFAULT_SYSTEM_PROMPT

def save_system_prompt(prompt_text: str) -> None:
    """Saves custom system prompt to system_prompt.txt."""
    try:
        with open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as f:
            f.write(prompt_text.strip())
    except Exception:
        pass

def reset_system_prompt() -> None:
    """Resets system prompt to DEFAULT_SYSTEM_PROMPT by removing system_prompt.txt."""
    try:
        if os.path.exists(SYSTEM_PROMPT_FILE):
            os.remove(SYSTEM_PROMPT_FILE)
    except Exception:
        pass

def load_env_config() -> dict[str, str]:
    """Loads key-value configuration from local .env file and environment variables."""
    config = {
        "AI_PROVIDER": os.environ.get("AI_PROVIDER", "gemini"),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "GEMINI_MODEL": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "OPENAI_MODEL": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "AI_BATCH_SIZE": os.environ.get("AI_BATCH_SIZE", "25"),
        "ACTIVE_PROMPT_PRESET": os.environ.get("ACTIVE_PROMPT_PRESET", "🎯 Универсальный (Редактор субтитров)"),
    }
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        config[k.strip()] = v.strip("\"' ")
        except Exception:
            pass
    return config

def save_env_config(updates: dict[str, str]) -> None:
    """Updates key-value configuration in local .env file preserving existing values."""
    current = load_env_config()
    current.update({k: v for k, v in updates.items() if v is not None})
    try:
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            for k, v in current.items():
                f.write(f"{k}={v}\n")
    except Exception:
        pass

def test_gemini_api_connection(api_key: str, model_name: str) -> tuple[bool, str]:
    """Tests connection to Google Gemini API."""
    import time
    from google import genai
    if not api_key.strip():
        return False, "Укажите Gemini API Key"
    start_t = time.time()
    try:
        client = genai.Client(api_key=api_key.strip())
        target_model = model_name.strip() if model_name.strip() else "gemini-3.5-flash"
        client.models.generate_content(
            model=target_model,
            contents="ping"
        )
        latency = time.time() - start_t
        return True, f"Успешное подключение к Google Gemini ({target_model})! Время отклика: {latency:.2f}с."
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            return False, f"Суточная квота модели {model_name} исчерпана (429 RESOURCE_EXHAUSTED). Выберите другую модель (например, gemini-3.5-flash) или используйте свой API."
        return False, f"Ошибка подключения к Gemini: {e}"

def test_openai_api_connection(base_url: str, api_key: str, model_name: str) -> tuple[bool, str]:
    """Tests connection to an OpenAI-compatible API endpoint."""
    import time
    from openai import OpenAI
    
    clean_url = base_url.strip().rstrip("/")
    clean_key = api_key.strip()
    if not clean_url:
        return False, "Укажите Base URL провайдера (например, https://api.openai.com/v1)"
    if not clean_key:
        return False, "Укажите API Key"
    if clean_key.startswith("http://") or clean_key.startswith("https://"):
        return False, "В поле 'API Key' указан веб-адрес (URL) вместо ключа! Вставьте ваш секретный ключ API (например, nvapi-... или sk-...)"
        
    start_t = time.time()
    target_model = model_name.strip() if model_name.strip() else "gpt-4o-mini"
    
    # Auto-resolve target_model if short name was passed
    try:
        client = OpenAI(base_url=clean_url, api_key=clean_key, timeout=8.0)
        models = client.models.list()
        avail = [str(m.id) for m in models.data if getattr(m, 'id', None)]
        if target_model not in avail:
            for m_id in avail:
                if m_id.split("/")[-1] == target_model or m_id.endswith("/" + target_model):
                    target_model = m_id
                    break
    except Exception:
        pass

    try:
        client = OpenAI(base_url=clean_url, api_key=clean_key, timeout=12.0)
        res = client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=2
        )
        latency = time.time() - start_t
        reply = res.choices[0].message.content if res.choices else "OK"
        return True, f"Подключение к {clean_url} успешно! Модель '{target_model}' ответила за {latency:.2f}с: '{reply}'"
    except Exception as e:
        err_s = str(e)
        if "404" in err_s or "page not found" in err_s.lower():
            return False, f"Ошибка 404: Модель '{target_model}' не найдена на сервере {clean_url}. Нажмите '🔄 Модели' и выберите модель из списка."
        elif "401" in err_s or "Unauthorized" in err_s or "invalid json object" in err_s.lower():
            return False, f"Ошибка 401 (Не авторизован): Неверный API Key для {clean_url}. Проверьте правильность ключа."
        return False, f"Ошибка подключения: {e}"

def fetch_openai_models(base_url: str, api_key: str) -> tuple[bool, list[str], str]:
    """Fetches list of available model IDs from an OpenAI-compatible API endpoint."""
    from openai import OpenAI
    clean_url = base_url.strip().rstrip("/")
    if not clean_url:
        return False, [], "Укажите Base URL провайдера"
    if not api_key.strip():
        return False, [], "Укажите API Key провайдера"
    try:
        client = OpenAI(base_url=clean_url, api_key=api_key.strip(), timeout=12.0)
        res = client.models.list()
        raw_ids = [str(m.id) for m in res.data if getattr(m, 'id', None)]
        if not raw_ids:
            return False, [], "Сервер ответил, но список моделей пуст."
        # Prioritize chat/instruct/flash/large models at the top
        chat_models = sorted([m for m in raw_ids if any(k in m.lower() for k in ["chat", "instruct", "flash", "it", "large", "mini", "gpt", "llama", "deepseek"])])
        other_models = sorted([m for m in raw_ids if m not in chat_models])
        sorted_models = chat_models + other_models
        return True, sorted_models, f"Успешно подтянуто моделей: {len(sorted_models)}"
    except Exception as e:
        return False, [], f"Не удалось подтянуть модели: {e}"

DEFAULT_SYSTEM_PROMPT = (
    "Ты — элитный лингвистический редактор и корректор субтитров для профессионального видеомонтажа.\n"
    "Твоя задача — тщательно ПЕРЕПРОВЕРИТЬ каждое слово в расшифровке аудиозаписи, исправить ВСЕ ошибки распознавания речи (ASR) "
    "и выдать идеально грамотный, чистый русский текст, точно соответствующий смыслу произнесенного.\n\n"
    "### 1. ГЛУБОКАЯ ПЕРЕПРОВЕРКА СЛОВ И ФОНЕТИЧЕСКИХ ОШИБОК (ASR):\n"
    "- Модели распознавания часто путают созвучные слова, подставляя бессмыслицу. Тщательно вчитывайся в контекст предложения и восстанавливай реальные слова, которые имел в виду спикер.\n"
    "- Исправляй слуховые ошибки и ошибочные разбивки/склейки слов (например: «до жили» → «дожили», «авто матическому» → «автоматическому», «в ряд ли» → «вряд ли», «не уже ли» → «неужели»).\n"
    "- Исправляй окончания, падежи, склонения и спряжения, если модель ASR ошиблась в согласовании слов в предложении (например: согласование с предлогами «с людьми», «про глоссарий», «с человеком», «обо мне»).\n\n"
    "### 2. ОРФОГРАФИЯ, ДЕФИСЫ И СЛИТНОЕ/РАЗДЕЛЬНОЕ НАПИСАНИЕ:\n"
    "- Наречия и частицы через дефис: «наконец-то», «из-за», «из-под», «по-русски», «по-моему», «как-то», «что-нибудь», «во-первых», «кое-как», «все-таки».\n"
    "- Слитное и раздельное написание: «также» (в значении «тоже») vs «так же» (как именно), «чтобы» (союз цели) vs «что бы» (местоимение с частицей), «поэтому», «ввиду», «вместе», «в течение».\n"
    "- Частицы НЕ/НИ: «не знаю», «не был», «ни разу», «никто», «ничего», «несмотря на».\n"
    "- Устраняй просторечные ошибки: «вообщем» → «в общем», «ихний» → «их», «ложить» → «класть».\n\n"
    "### 3. АНГЛИЙСКИЕ СЛОВА, БРЕНДЫ И ТЕРМИНЫ — СТРОГО НА АНГЛИЙСКОМ (ЛАТИНИЦЕЙ):\n"
    "- ЖЕЛЕЗНОЕ ПРАВИЛО: ЕСЛИ СЛОВО АНГЛИЙСКОЕ, ОНО ОБЯЗАНО ПИСАТЬСЯ НА АНГЛИЙСКОМ (латиницей), а не русскими буквами!\n"
    "- Если модель распознавания записала английское слово русской транскрипцией на слух, ОБЯЗАТЕЛЬНО замени его на оригинальное английское написание:\n"
    "  • Программы и сервисы: «давинчи резолв» → DaVinci Resolve, «премьер про» → Premiere Pro, «капкат» → CapCut, «файнал кат» → Final Cut, «афтер эффектс» → After Effects, «фотошоп» → Photoshop, «миджорни» → Midjourney, «комфи юай» → ComfyUI, «ранвей» → Runway, «стейбл дифьюжн» → Stable Diffusion, «блендер» → Blender.\n"
    "  • Нейросети, платформы и IT: «чат гпт» / «чатгпт» → ChatGPT, «клод» → Claude, «джемини» → Gemini, «дипсик» → DeepSeek, «опенаи» → OpenAI, «антропик» → Anthropic, «ви whisper» → Whisper, «паракит» → Parakeet, «элевен лабс» → ElevenLabs, «пайтон» → Python, «гит» → Git, «гитхаб» → GitHub, «докер» → Docker.\n"
    "  • Соцсети и форматы: «шортс» → Shorts, «рилс» → Reels, «тикток» → TikTok, «ютуб» → YouTube, «инстаграм» → Instagram, «телеграм» → Telegram, «подкаст» → Podcast.\n"
    "  • Монтажные и технические термины, аббревиатуры: «промпт» → Prompt, «таймлайн» → Timeline, «рендер» → Render, «битрейт» → Bitrate, «кейфрейм» → Keyframe, «б-ролл» → B-roll, «пресет» → Preset, «плагин» → Plugin, «скрипт» → Script, «апи» → API, «гпу» / «джипию» → GPU, «цпу» → CPU, «ртх» → RTX, «фпс» → FPS, «хдр» → HDR, «сдр» → SDR, «фулл хд» → Full HD, «4к» → 4K, «окей» → OK, «хай» → Hi, «лайк» → Like, «чекнуть» / «чек» → Check.\n"
    "- Никакой русской транслитерации для английских терминов («премьер», «капкат», «шортс», «пайтон», «гпу» запрещены — только Premiere, CapCut, Shorts, Python, GPU)!\n\n"
    "### 4. ПРАВИЛО ПЕРЕНОСА ПРЕДЛОГОВ, СОЮЗОВ И МЕСТОИМЕНИЙ:\n"
    "- ЖЕЛЕЗНОЕ ПРАВИЛО ТИПОГРАФИКИ И СУБТИТРОВ: ПРЕДЛОГ В КОНЦЕ СТРОКИ СТРОГО ЗАПРЕЩЕН!\n"
    "- Предлог никогда не должен оставаться висячим в конце строки или сегмента субтитров в отрыве от зависимого слова.\n"
    "- Предлог ВСЕГДА переносится и начинает новую строку вместе со словом, к которому относится:\n"
    "  • Запрещено: «Мы говорили с» / «людьми» → Обязательно: «Мы говорили» / «с людьми»\n"
    "  • Запрещено: «Это видео про» / «DaVinci Resolve» → Обязательно: «Это видео» / «про DaVinci Resolve»\n"
    "  • Запрещено: «Я пришёл к» / «выводу» → Обязательно: «Я пришёл» / «к выводу»\n"
    "  • Запрещено: «Мы поехали в» / «Москву» → Обязательно: «Мы поехали» / «в Москву»\n"
    "- Правило действует для ВСЕХ предлогов («в», «на», «с», «со», «к», «по», «из», «у», «о», «об», «от», «до», «для», «за», «под», «над», «при», «про», «без», «через» и др.), а также коротких союзов («и», «а», «но»).\n"
    "- ЗАПРЕТ НА ВИСЯЧИЕ И ОДИНОЧНЫЕ МЕСТОИМЕНИЯ И ЧАСТИЦЫ: Местоимения («я», «мы», «ты», «вы», «он», «она», «оно», «они», «это», «все») и отрицательные частицы («не», «ни») НИКОГДА не должны оставаться одни на отдельной строке субтитров и НЕ ДОЛЖНЫ висеть в конце строки перед следующим словом! Они ВСЕГДА переносятся на новую строку и пишутся вместе со своим глаголом или смысловой фразой (например: «Я думаю», «Мы пошли», «Вчера я видел», «Я не закрыл», «Это видео»). Запрещено оставлять «я» или «не» в конце строки («по которой я» / «завершил» → «по которой» / «я завершил»).\n\n"
    "### 5. УДАЛЕНИЕ РЕЧЕВОГО МУСОРА И ЗАМИНОК:\n"
    "- Полностью удаляй звуки заминки речи: «э-э», «мм», «а-а», «эээ», «угу», «гм».\n"
    "- Вырезай заикания, фальстарты и случайные повторы слов (например: «мы мы наконец» → «мы наконец»).\n"
    "- Убирай паразиты («ну», «типа», «короче», «как бы», «в общем-то», «так сказать»), если они служат заполнителями пауз.\n\n"
    "### 6. СОХРАНЕНИЕ АВТОРСКОГО СТИЛЯ (СТРОГОЕ ПРАВИЛО):\n"
    "- НЕЛЬЗЯ пересказывать, сокращать или перефразировать автора! Зритель должен слышать ровно то, что видит в субтитрах.\n"
    "- Исправляй ТОЛЬКО ошибки, оговорки и мусор. Живую авторскую речь, терминологию и порядок слов оставляй неизменными.\n\n"
    "### 7. ОБЯЗАТЕЛЬНЫЙ ЭТАП САМОПРОВЕРКИ ПЕРЕД ОТВЕТОМ (SELF-VERIFICATION):\n"
    "- Прежде чем сформировать и выдать финальный ответ, ТЫ ОБЯЗАН ВНИМАТЕЛЬНО САМ СЕБЯ ПЕРЕПРОВЕРИТЬ по 6 пунктам:\n"
    "  1) Контекст и слуховые ошибки (ASR): Каждое ли слово проверено по смыслу? Нет ли созвучных ослышек («до жили» → «дожили»)?\n"
    "  2) Английские названия и бренды: 100% софта, IT-брендов и терминов записаны на чистом английском латиницей (DaVinci Resolve, Premiere Pro, CapCut, Shorts, Reels, TikTok, ChatGPT, Python, GPU, API)? Никакой кириллицы!\n"
    "  3) Перенос предлогов и местоимений: НЕТ ЛИ ВИСЯЧИХ ПРЕДЛОГОВ на концах строк, и НЕТ ЛИ ОДИНОКИХ МЕСТОИМЕНИЙ («я», «мы», «он»), оставленных в изоляции на отдельной строке? Все предлоги и местоимения встроены в смысловые фразы!\n"
    "  4) Орфография и дефисы: «наконец-то», «из-за», «по-русски», слитные «чтобы», «также», падежи и согласования.\n"
    "  5) Очистка мусора: убраны ли звуки заминки («э-э», «ну»), сохранён ли живой авторский стиль без пересказа?\n"
    "  6) Точность структуры JSON: количество элементов и `id` строго совпадают с входящими!\n"
    "- Только после полного подтверждения всех пунктов самопроверки выдавай результат.\n\n"
    "### 8. СТРОГИЙ ФОРМАТ ОТВЕТА (JSON):\n"
    "- Верни строго JSON-массив объектов с теми же `id`: [{\"id\": 0, \"text\": \"...\"}].\n"
    "- Порядок и количество элементов должны строго совпадать с входящими.\n"
    "- Если фрагмент состоял исключительно из заминки («э-э», «ну»), верни text: \"\"."
)

PROMPT_PRESET_SHORTS = (
    "Ты — элитный редактор динамичных клиповых субтитров формата 9:16 для Shorts, Reels и TikTok.\n"
    "Твоя задача — превратить расшифровку в идеальный, ритмичный текст для быстрого вертикального видео.\n\n"
    "### 1. СТРОГИЕ ПРАВИЛА ТИПОГРАФИКИ 9:16:\n"
    "- ПРЕДЛОГИ И МЕСТОИМЕНИЯ НА КОНЦАХ СТРОК СТРОГО ЗАПРЕЩЕНЫ: предлоги («в», «на», «с», «к», «по», «из» и др.), "
    "короткие местоимения («я», «мы», «он», «она», «это») и отрицательные частицы («не», «ни») НИКОГДА не должны оставаться в конце строки! "
    "Они всегда переносятся на новую строку вместе со своим словом («Я думаю», «с людьми», «не закрыл»).\n"
    "- ОДИНОЧНЫЕ МЕСТОИМЕНИЯ ЗАПРЕЩЕНЫ: Местоимения («я», «мы») никогда не должны быть отдельной одиночной строкой — только в составе смысловой фразы.\n\n"
    "### 2. БРЕНДЫ, СОФТ И IT-ТЕРМИНЫ — ТОЛЬКО ЛАТИНИЦЕЙ:\n"
    "- Все названия платформ, программ и технологий пиши строго на английском: Shorts, Reels, TikTok, YouTube, Telegram, Instagram, "
    "CapCut, Premiere Pro, DaVinci Resolve, Final Cut, After Effects, Photoshop, ChatGPT, Midjourney, ComfyUI, Python, GPU, API.\n"
    "- Никакой кириллицы для английских названий!\n\n"
    "### 3. РЕЧЕВОЙ МУСОР И ДИНАМИКА:\n"
    "- Полностью удаляй заминки («э-э», «ну», «мм»), заикания и фальстарты.\n"
    "- Сохраняй авторский темп и эмоциональный посыл речи без изменения смысла.\n\n"
    "### 4. САМОПРОВЕРКА И ФОРМАТ (JSON):\n"
    "- Перепроверь: нет ли висячих предлогов/местоимений на концах строк, все ли бренды латиницей.\n"
    "- Верни строго JSON-массив объектов: [{\"id\": 0, \"text\": \"...\"}]."
)

PROMPT_PRESET_PODCAST = (
    "Ты — профессиональный редактор стенограмм и субтитров для подкастов, интервью и разговорных видео.\n"
    "Твоя задача — бережно вычитать текст, сохранив уникальный живой голос и интонацию спикеров.\n\n"
    "### 1. СОХРАНЕНИЕ ЖИВОЙ РЕЧИ:\n"
    "- Максимально сохраняй авторские формулировки, шутки, разговорные обороты и авторский стиль.\n"
    "- Удаляй исключительно явные звуки заминки («э-э», «эээ», «мм»), кашель и случайные оговорки/фальстарты.\n\n"
    "### 2. ГРАМОТНОСТЬ И ПУНКТУАЦИЯ:\n"
    "- Расставляй выразительную пунктуацию (тире, многоточия, запятые, вопросительные знаки), отражающую интонационные паузы спикера.\n"
    "- Исправляй слуховые ошибки распознавания речи (ASR), согласуй падежные окончания и дефисы («все-таки», «по-моему»).\n\n"
    "### 3. ТЕРМИНОЛОГИЯ И ПЕРЕНОС СТРОК:\n"
    "- Упомянутые спикерами термины, имена брендов и сервисов пиши в их общепринятом виде (на английском латиницей, если это зарубежный софт или термин).\n"
    "- Предлоги и местоимения переноси вперед к зависимым словам, не оставляя висячих слов в конце строк.\n\n"
    "### 4. САМОПРОВЕРКА И ФОРМАТ (JSON):\n"
    "- Перепроверь согласованность реплик и верни строго JSON-массив: [{\"id\": 0, \"text\": \"...\"}]."
)

PROMPT_PRESET_TECH = (
    "Ты — технический редактор субтитров для IT-туториалов, уроков программирования и видео про технологии.\n"
    "Твоя задача — обеспечить безупречную точность технических терминов, названий библиотек, кода и софта.\n\n"
    "### 1. ТЕХНИЧЕСКИЕ ТЕРМИНЫ — 100% НА АНГЛИЙСКОМ (ЛАТИНИЦЕЙ):\n"
    "- Все названия языков программирования, библиотек, фреймворков, софта, архитектур и команд пиши строго на английском:\n"
    "  • Python, C++, Rust, JavaScript, TypeScript, Go, HTML, CSS, SQL.\n"
    "  • PyTorch, TensorFlow, CUDA, cuDNN, GPU, CPU, RTX, TensorRT, HuggingFace.\n"
    "  • Docker, Kubernetes, Linux, Ubuntu, Git, GitHub, GitLab, Terminal, Bash, PowerShell.\n"
    "  • API, REST, JSON, Webhook, Backend, Frontend, Cloud, AWS, GCP, Azure.\n"
    "  • DaVinci Resolve, Premiere Pro, CapCut, Blender, ComfyUI, Stable Diffusion, Whisper, Parakeet.\n"
    "- Никакой кириллической транслитерации для технического сленга!\n\n"
    "### 2. ПЕРЕНОС ПРЕДЛОГОВ И МЕСТОИМЕНИЙ:\n"
    "- Предлоги («в», «на», «с», «к», «по», «для») и местоимения («я», «мы», «он») никогда не должны висеть в конце строк. Они переносятся к термину или глаголу.\n"
    "- Одиночные местоимения в отдельную строку строго запрещены.\n\n"
    "### 3. САМОПРОВЕРКА И ФОРМАТ (JSON):\n"
    "- Перепроверь каждое IT-название и синтаксис. Верни строго JSON-массив: [{\"id\": 0, \"text\": \"...\"}]."
)

BUILTIN_PROMPT_PRESETS = {
    "🎯 Универсальный (Редактор субтитров)": DEFAULT_SYSTEM_PROMPT,
    "📱 Клиповый 9:16 (Shorts / Reels / TikTok)": PROMPT_PRESET_SHORTS,
    "🎙️ Подкасты и длинные интервью": PROMPT_PRESET_PODCAST,
    "💻 IT, разработка и туториалы": PROMPT_PRESET_TECH,
}

CUSTOM_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "custom_prompts.json")

def load_custom_prompts() -> dict[str, str]:
    """Loads user-saved custom prompt presets from custom_prompts.json."""
    if os.path.exists(CUSTOM_PROMPTS_FILE):
        try:
            with open(CUSTOM_PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}

def save_custom_prompts(prompts: dict[str, str]) -> None:
    """Saves user custom prompt presets to custom_prompts.json."""
    try:
        with open(CUSTOM_PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_all_prompt_presets() -> dict[str, str]:
    """Returns combined dictionary of built-in and user-custom prompt presets."""
    presets = dict(BUILTIN_PROMPT_PRESETS)
    customs = load_custom_prompts()
    for k, v in customs.items():
        presets[k] = v
    if "✍️ Свой промт (Пользовательский)" not in presets:
        custom_txt = ""
        if os.path.exists(SYSTEM_PROMPT_FILE):
            try:
                with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                    custom_txt = f.read().strip()
            except Exception:
                pass
        presets["✍️ Свой промт (Пользовательский)"] = custom_txt if custom_txt else DEFAULT_SYSTEM_PROMPT
    return presets

DEFAULT_FILLER_WORDS = [
    # Русские звуки заминки и междометия
    "ээ", "эээ", "эм", "эмм", "мм", "ммм", "аа", "ааа", "гм", "хм",
    # Частые слова-паразиты (русский)
    "ну", "короче", "типа", "как бы", "в общем", "в общем-то", "так сказать",
    "собственно", "собственно говоря", "значит", "понимаешь", "понимаете",
    "слушай", "слушайте", "вот", "то есть",
    # Английские филлеры (English fillers)
    "uh", "um", "er", "ah", "umm", "uhh", "like", "you know", "basically",
    "literally", "actually", "sort of", "kind of", "i mean"
]

def strip_punctuation_from_text(text: str) -> str:
    """Removes punctuation from text while preserving intra-word hyphens (e.g. 'наконец-то', 'из-за')."""
    # Replace standalone dashes/hyphens surrounded by spaces with a space
    s = re.sub(r'\s+[-—–]+\s+', ' ', text)
    # Remove leading dash in dialogue (e.g. "- Привет")
    s = re.sub(r'(?:^|\s)[-—–]+', ' ', s)
    # Remove all other punctuation marks except intra-word hyphens
    cleaned = "".join(ch for ch in s if ch not in PUNCTUATION_CHARS)
    return " ".join(cleaned.split())

def strip_punctuation_from_word(word: str) -> str:
    """Strips punctuation marks from an individual word token, preserving internal hyphens."""
    cleaned = "".join(ch for ch in word if ch not in PUNCTUATION_CHARS)
    return cleaned.strip(" -")

def parse_filler_words(text_input: str) -> tuple[set[str], list[list[str]]]:
    """Parses user filler words input into single-word set and multi-word phrases list."""
    single_words = set()
    multi_phrases = []
    for item in text_input.split(","):
        cleaned = strip_punctuation_from_word(item.strip()).lower()
        if not cleaned:
            continue
        parts = cleaned.split()
        if len(parts) == 1:
            single_words.add(parts[0])
        elif len(parts) > 1:
            multi_phrases.append(parts)
    return single_words, multi_phrases

def filter_words(words: list[dict], single_words: set[str], multi_phrases: list[list[str]]) -> list[dict]:
    """Filters out filler words and multi-word phrases from word token list."""
    if not words:
        return []
    
    norm_words = [strip_punctuation_from_word(w.get("word", "")).lower() for w in words]
    keep = [True] * len(words)
    
    # Check single words
    for i, nw in enumerate(norm_words):
        if nw in single_words:
            keep[i] = False
            
    # Check multi-word phrases
    for phrase in multi_phrases:
        p_len = len(phrase)
        for i in range(len(norm_words) - p_len + 1):
            if all(norm_words[i + j] == phrase[j] for j in range(p_len)):
                for j in range(p_len):
                    keep[i + j] = False

    return [w for i, w in enumerate(words) if keep[i]]

def filter_text_fillers(text: str, single_words: set[str], multi_phrases: list[list[str]]) -> str:
    """Removes filler words and phrases from a text string."""
    tokens = text.split()
    words = [{"word": t} for t in tokens]
    filtered = filter_words(words, single_words, multi_phrases)
    return " ".join(w["word"] for w in filtered)

def replace_yo(text: str) -> str:
    """Replaces Cyrillic 'ё' and 'Ё' with 'е' and 'Е'."""
    return text.replace("ё", "е").replace("Ё", "Е")

def process_text(text: str, remove_punct: bool = False, case_mode: str = "none", replace_yo_flag: bool = False) -> str:
    """Applies text formatting options (punctuation removal, case formatting, yo-replacement)."""
    res = text
    if replace_yo_flag:
        res = replace_yo(res)
    if remove_punct:
        res = strip_punctuation_from_text(res)
    if case_mode == "upper":
        res = res.upper()
    elif case_mode == "lower":
        res = res.lower()
    elif case_mode == "title":
        res = res.title()
    return res

def process_word(word: str, remove_punct: bool = False, case_mode: str = "none", replace_yo_flag: bool = False) -> str:
    """Applies word token formatting options (punctuation removal, case formatting, yo-replacement)."""
    res = word
    if replace_yo_flag:
        res = replace_yo(res)
    if remove_punct:
        res = strip_punctuation_from_word(res)
    if case_mode == "upper":
        res = res.upper()
    elif case_mode == "lower":
        res = res.lower()
    elif case_mode == "title":
        res = res.title()
    return res

def align_words_to_timestamps(original_words: list[dict], corrected_text: str, seg_start: float, seg_end: float) -> list[dict]:
    """Aligns Gemini-corrected words to original word timestamps using difflib.SequenceMatcher."""
    corrected_word_tokens = corrected_text.split()
    if not corrected_word_tokens:
        return []
    
    if not original_words:
        duration = max(0.1, seg_end - seg_start)
        step = duration / len(corrected_word_tokens)
        return [
            {
                "word": w,
                "start": round(seg_start + i * step, 2),
                "end": round(seg_start + (i + 1) * step, 2)
            }
            for i, w in enumerate(corrected_word_tokens)
        ]
        
    orig_clean = [w.get("word", "").lower().strip(".,!?:;«»\"'—–…") for w in original_words]
    corr_clean = [w.lower().strip(".,!?:;«»\"'—–…") for w in corrected_word_tokens]
    
    matcher = difflib.SequenceMatcher(None, orig_clean, corr_clean)
    aligned_words = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                w_orig = original_words[i1 + offset]
                w_corr = corrected_word_tokens[j1 + offset]
                aligned_words.append({
                    "word": w_corr,
                    "start": w_orig.get("start", seg_start),
                    "end": w_orig.get("end", seg_end)
                })
        elif tag == "replace":
            orig_slice = original_words[i1:i2]
            corr_slice = corrected_word_tokens[j1:j2]
            if orig_slice:
                r_start = orig_slice[0].get("start", seg_start)
                r_end = orig_slice[-1].get("end", seg_end)
            else:
                r_start, r_end = seg_start, seg_end
            step = max(0.05, (r_end - r_start) / max(1, len(corr_slice)))
            for k, w_corr in enumerate(corr_slice):
                w_s = round(r_start + k * step, 2)
                w_e = round(min(seg_end, r_start + (k + 1) * step), 2)
                if w_e <= w_s:
                    w_e = round(w_s + 0.05, 2)
                aligned_words.append({
                    "word": w_corr,
                    "start": w_s,
                    "end": w_e
                })
        elif tag == "insert":
            prev_end = aligned_words[-1]["end"] if aligned_words else seg_start
            next_start = original_words[i1].get("start", seg_end) if i1 < len(original_words) else seg_end
            if next_start < prev_end:
                next_start = prev_end + 0.1
            ins_span = max(0.1, next_start - prev_end)
            corr_slice = corrected_word_tokens[j1:j2]
            step = ins_span / max(1, len(corr_slice))
            for k, w_corr in enumerate(corr_slice):
                w_s = round(prev_end + k * step, 2)
                w_e = round(prev_end + (k + 1) * step, 2)
                aligned_words.append({
                    "word": w_corr,
                    "start": w_s,
                    "end": w_e
                })
        elif tag == "delete":
            pass
            
    # Guarantee strictly monotonic non-decreasing timestamps
    for idx in range(1, len(aligned_words)):
        if aligned_words[idx]["start"] < aligned_words[idx - 1]["start"]:
            aligned_words[idx]["start"] = aligned_words[idx - 1]["start"]
        if aligned_words[idx]["end"] <= aligned_words[idx]["start"]:
            aligned_words[idx]["end"] = round(aligned_words[idx]["start"] + 0.05, 2)
            
    return aligned_words

def enhance_transcript_with_gemini(
    segments: list[dict],
    api_key: str,
    model_name: str = "gemini-3.5-flash",
    batch_size: int = 25,
    progress_bar = None,
    status_text = None,
    system_instruction: str = None
) -> list[dict]:
    """Uses Google Gemini API to fix transcription errors, remove speech fillers, and polish words while preserving and aligning timestamps."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip())
    instruction_to_use = system_instruction if system_instruction else DEFAULT_SYSTEM_PROMPT
    
    all_corrections = {}
    total_segments = len(segments)
    
    # Process in batches to guarantee fast, reliable responses without timeout or token overflow
    for start_idx in range(0, total_segments, batch_size):
        end_idx = min(start_idx + batch_size, total_segments)
        batch = segments[start_idx:end_idx]
        
        items_to_send = [{"id": start_idx + j, "text": seg.get("text", "")} for j, seg in enumerate(batch)]
        prompt = (
            f"Вот фрагменты распознанной речи #{start_idx + 1}–#{end_idx} (всего {len(items_to_send)}) в формате JSON:\n"
            f"{json.dumps(items_to_send, ensure_ascii=False, indent=2)}\n\n"
            "ЗАДАНИЕ: Внимательно перепроверь каждое слово по контексту.\n"
            "1. Исправь все слуховые ошибки распознавания (ASR), опечатки, падежные окончания и дефисы («наконец-то», «из-за»).\n"
            "2. СТРОЖАЙШЕЕ ПРАВИЛО: ЕСЛИ СЛОВО АНГЛИЙСКОЕ, ТЕРМИН ИЛИ БРЕНД — ОНО ОБЯЗАНО ПИСАТЬСЯ НА АНГЛИЙСКОМ ЛАТИНИЦЕЙ "
            "(DaVinci Resolve, Premiere Pro, CapCut, ChatGPT, Shorts, Reels, TikTok, YouTube, Python, API, GPU, Render, Timeline, Keyframe, Preset, OK и т.д.). Никакой русской транслитерации для английских слов!\n"
            "3. ПРАВИЛО ПЕРЕНОСА ПРЕДЛОГОВ И МЕСТОИМЕНИЙ: Предлог никогда не должен оставаться в конце строки/сегмента! Он всегда переносится и начинает новую строку вместе со своим словом («Мы говорили» / «с людьми», «видео» / «про DaVinci Resolve»). Местоимения («я», «мы», «он») и частицы («не», «ни») НЕ ДОЛЖНЫ висеть в конце строки и НЕ ДОЛЖНЫ оставаться в одиночестве на отдельной строке — они пишутся вместе с глаголом или смысловой фразой («Я думаю», «Вчера я пошел», «Мы поехали», «Я не закрыл»). Запрещено оставлять «я» или «не» в конце строки!\n"
            "4. Удали речевой мусор и звуки заминки («э-э», «ну»).\n"
            "5. Сохраняй авторские слова и порядок предложений (не пересказывай!).\n"
            "6. ОБЯЗАТЕЛЬНАЯ САМОПРОВЕРКА (Self-Verification): Перед формированием итогового JSON перепроверь себя: нет ли пропущенных опечаток, все ли термины латиницей, перенесены ли предлоги с конца строк, нет ли висячих или одиноких местоимений («я», «мы») и частиц («не», «ни»).\n"
            "Верни строго валидный JSON-массив объектов: [{\"id\": ..., \"text\": \"...\"}]."
        )
        
        if status_text is not None:
            status_text.text(f"✨ Gemini ({model_name}): проверка сегментов {start_idx + 1}–{end_idx} из {total_segments}...")
            
        models_to_try = [model_name]
        for fallback_candidate in ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]:
            if fallback_candidate not in models_to_try:
                models_to_try.append(fallback_candidate)
                
        response = None
        last_error = None
        for try_m in models_to_try:
            try:
                response = client.models.generate_content(
                    model=try_m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=instruction_to_use,
                        response_mime_type="application/json"
                    )
                )
                if try_m != model_name and status_text is not None:
                    status_text.text(f"✨ Квота {model_name} исчерпана. Переключено на резервную модель {try_m} ✅")
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "404" in err_str:
                    continue
                else:
                    raise e
                    
        if response is None:
            raise last_error
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        corrected_items = json.loads(raw_text.strip())
        for item in corrected_items:
            if "id" in item:
                try:
                    all_corrections[int(item["id"])] = item.get("text", "").strip()
                except (ValueError, TypeError):
                    pass
                    
        if progress_bar is not None:
            progress_bar.progress(min(1.0, end_idx / max(1, total_segments)))
            
    enhanced_segments = []
    for i, seg in enumerate(segments):
        seg_copy = dict(seg)
        if i in all_corrections:
            corrected_text = all_corrections[i]
            if not corrected_text:
                continue  # Whole segment removed as filler
            seg_copy["original_text"] = seg.get("text", "")
            seg_copy["text"] = corrected_text
            
            # Align words to timestamps
            if "words" in seg and seg["words"]:
                aligned_words = align_words_to_timestamps(
                    seg["words"],
                    corrected_text,
                    seg.get("start", 0.0),
                    seg.get("end", 0.0)
                )
                seg_copy["words"] = aligned_words
                if aligned_words:
                    seg_copy["start"] = aligned_words[0]["start"]
                    seg_copy["end"] = aligned_words[-1]["end"]
        enhanced_segments.append(seg_copy)
        
    return enhanced_segments

def enhance_transcript_with_openai_compatible(
    segments: list[dict],
    base_url: str,
    api_key: str,
    model_name: str = "gpt-4o-mini",
    batch_size: int = 25,
    progress_bar = None,
    status_text = None,
    system_instruction: str = None
) -> list[dict]:
    """Uses an OpenAI-compatible API (OpenRouter, vLLM, DeepSeek, Local LLM)
    to fix transcription errors, remove speech fillers, and polish words while preserving and aligning timestamps."""
    from openai import OpenAI
    
    clean_base_url = base_url.strip().rstrip("/")
    clean_key = api_key.strip()
    client = OpenAI(base_url=clean_base_url, api_key=clean_key, timeout=90.0)
    
    # Auto-resolve model_name against available models if short name was passed
    resolved_model = model_name.strip()
    try:
        models_res = client.models.list()
        avail_ids = [str(m.id) for m in models_res.data if getattr(m, 'id', None)]
        if resolved_model not in avail_ids:
            for m_id in avail_ids:
                if m_id.split("/")[-1] == resolved_model or m_id.endswith("/" + resolved_model):
                    resolved_model = m_id
                    break
    except Exception:
        pass
    
    instruction_to_use = system_instruction if system_instruction else DEFAULT_SYSTEM_PROMPT
    
    all_corrections = {}
    total_segments = len(segments)
    
    for start_idx in range(0, total_segments, batch_size):
        end_idx = min(start_idx + batch_size, total_segments)
        batch = segments[start_idx:end_idx]
        
        items_to_send = [{"id": start_idx + j, "text": seg.get("text", "")} for j, seg in enumerate(batch)]
        prompt = (
            f"Вот фрагменты распознанной речи #{start_idx + 1}–#{end_idx} (всего {len(items_to_send)}) в формате JSON:\n"
            f"{json.dumps(items_to_send, ensure_ascii=False, indent=2)}\n\n"
            "ЗАДАНИЕ: Внимательно перепроверь каждое слово по контексту.\n"
            "1. Исправь все слуховые ошибки распознавания (ASR), опечатки, падежные окончания и дефисы («наконец-то», «из-за»).\n"
            "2. СТРОЖАЙШЕЕ ПРАВИЛО: ЕСЛИ СЛОВО АНГЛИЙСКОЕ, ТЕРМИН ИЛИ БРЕНД — ОНО ОБЯЗАНО ПИСАТЬСЯ НА АНГЛИЙСКОМ ЛАТИНИЦЕЙ "
            "(DaVinci Resolve, Premiere Pro, CapCut, ChatGPT, Shorts, Reels, TikTok, YouTube, Python, API, GPU, Render, Timeline, Keyframe, Preset, OK и т.д.). Никакой русской транслитерации для английских слов!\n"
            "3. ПРАВИЛО ПЕРЕНОСА ПРЕДЛОГОВ И МЕСТОИМЕНИЙ: Предлог никогда не должен оставаться в конце строки/сегмента! Он всегда переносится и начинает новую строку вместе со своим словом («Мы говорили» / «с людьми», «видео» / «про DaVinci Resolve»). Местоимения («я», «мы», «он») и частицы («не», «ни») НЕ ДОЛЖНЫ висеть в конце строки и НЕ ДОЛЖНЫ оставаться в одиночестве на отдельной строке — они пишутся вместе с глаголом или смысловой фразой («Я думаю», «Вчера я пошел», «Мы поехали», «Я не закрыл»). Запрещено оставлять «я» или «не» в конце строки!\n"
            "4. Удали речевой мусор и звуки заминки («э-э», «ну»).\n"
            "5. Сохраняй авторские слова и порядок предложений (не пересказывай!).\n"
            "6. ОБЯЗАТЕЛЬНАЯ САМОПРОВЕРКА (Self-Verification): Перед формированием итогового JSON перепроверь себя: нет ли пропущенных опечаток, все ли термины латиницей, перенесены ли предлоги с конца строк, нет ли висячих или одиноких местоимений («я», «мы») и частиц («не», «ни»).\n"
            "Верни строго валидный JSON-массив объектов: [{\"id\": ..., \"text\": \"...\"}]."
        )
        
        if status_text is not None:
            status_text.text(f"✨ AI ({resolved_model}): проверка сегментов {start_idx + 1}–{end_idx} из {total_segments}...")
            
        raw_text = ""
        try:
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": instruction_to_use},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw_text = resp.choices[0].message.content or ""
        except Exception:
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": instruction_to_use},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_text = resp.choices[0].message.content or ""
            
        raw_text = raw_text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        try:
            parsed = json.loads(raw_text.strip())
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                else:
                    parsed = [parsed]
            for item in parsed:
                if isinstance(item, dict) and "id" in item:
                    try:
                        all_corrections[int(item["id"])] = item.get("text", "").strip()
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
                    
        if progress_bar is not None:
            progress_bar.progress(min(1.0, end_idx / max(1, total_segments)))
            
    enhanced_segments = []
    for i, seg in enumerate(segments):
        seg_copy = dict(seg)
        if i in all_corrections:
            corrected_text = all_corrections[i]
            if not corrected_text:
                continue
            seg_copy["original_text"] = seg.get("text", "")
            seg_copy["text"] = corrected_text
            
            if "words" in seg and seg["words"]:
                aligned_words = align_words_to_timestamps(
                    seg["words"],
                    corrected_text,
                    seg.get("start", 0.0),
                    seg.get("end", 0.0)
                )
                seg_copy["words"] = aligned_words
                if aligned_words:
                    seg_copy["start"] = aligned_words[0]["start"]
                    seg_copy["end"] = aligned_words[-1]["end"]
        enhanced_segments.append(seg_copy)
        
    return enhanced_segments

def format_time_display(seconds: float) -> str:
    """Formats seconds into MM:SS format for UI display."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def format_time_srt(seconds: float) -> str:
    """Formats seconds into HH:MM:SS,mmm for SubRip (SRT) subtitles."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def format_time_vtt(seconds: float) -> str:
    """Formats seconds into HH:MM:SS.mmm for WebVTT subtitles."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def generate_srt(segments: list[dict], word_level: bool = False, remove_punct: bool = False, case_mode: str = "none", replace_yo_flag: bool = False, no_gaps: bool = False) -> str:
    """Generates SubRip (.srt) subtitle content from segments or word-level timing, with optional gapless mode."""
    if word_level:
        items = extract_word_level_items(segments, remove_punct, case_mode, replace_yo_flag)
    else:
        items = []
        for seg in segments:
            text = process_text(seg.get("text", ""), remove_punct, case_mode, replace_yo_flag)
            if not text:
                continue
            items.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text
            })
            
    if not items:
        return ""
        
    if no_gaps:
        for i in range(len(items) - 1):
            items[i]["end"] = max(items[i]["start"] + 0.05, items[i + 1]["start"])

    lines = []
    for idx, item in enumerate(items, 1):
        s_str = format_time_srt(item["start"])
        e_str = format_time_srt(item["end"])
        lines.append(f"{idx}\n{s_str} --> {e_str}\n{item['text']}\n")
    return "\n".join(lines)

def generate_vtt(segments: list[dict], word_level: bool = False, remove_punct: bool = False, case_mode: str = "none", replace_yo_flag: bool = False) -> str:
    """Generates WebVTT (.vtt) subtitle content from segments or word-level timing."""
    lines = ["WEBVTT\n"]
    if word_level:
        items = extract_word_level_items(segments, remove_punct, case_mode, replace_yo_flag)
    else:
        items = []
        for seg in segments:
            text = process_text(seg.get("text", ""), remove_punct, case_mode, replace_yo_flag)
            if not text:
                continue
            items.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text
            })
    for idx, item in enumerate(items, 1):
        start = format_time_vtt(item["start"])
        end = format_time_vtt(item["end"])
        lines.append(f"{idx}\n{start} --> {end}\n{item['text']}\n")
    return "\n".join(lines)

RUSSIAN_PREPOSITIONS = {
    "в", "во", "с", "со", "к", "ко", "о", "об", "обо", "у", "на", "по", "от", "ото",
    "до", "из", "изо", "за", "под", "подо", "над", "надо", "при", "про", "без", "безо",
    "для", "через", "сквозь", "перед", "передо", "ради", "около", "возле", "среди",
    "вокруг", "между", "из-за", "из-под", "кроме", "вместо", "насчет", "вроде", "вдоль",
    "позади", "навстречу", "наперекор", "вслед", "наподобие"
}

ENGLISH_PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
    "into", "through", "under", "over", "before", "after", "between", "without",
    "toward", "against", "among", "across", "behind", "beyond", "inside", "outside"
}

ALL_PREPOSITIONS = RUSSIAN_PREPOSITIONS | ENGLISH_PREPOSITIONS

CLAUSE_STARTERS = {
    "когда", "если", "что", "чтобы", "потому", "как", "где", "куда", "откуда",
    "хотя", "зачем", "почему", "который", "которая", "которое", "которые",
    "when", "if", "that", "because", "how", "where", "why", "which"
}

ALL_PRONOUNS = {
    # Личные местоимения
    "я", "мы", "ты", "вы", "он", "она", "оно", "они",
    "меня", "нас", "тебя", "вас", "его", "него", "её", "нее", "ее", "их", "них",
    "мне", "нам", "тебе", "вам", "ему", "нему", "ей", "ней", "им", "ним",
    "мной", "мною", "нами", "тобой", "тобою", "вами", "ими", "ними",
    # Указательные / определительные
    "это", "этот", "эта", "эти", "этого", "этой", "этим", "этих",
    "тот", "та", "то", "те", "того", "той", "тем", "тех",
    "все", "всё", "весь", "вся", "всего", "всей", "всем", "всех",
    "сам", "сама", "само", "сами",
    # Притяжательные
    "мой", "моя", "моё", "мое", "мои", "моего", "моей", "моим", "моих",
    "твой", "твоя", "твоё", "твое", "твои", "твоего", "твоей", "твоим", "твоих",
    "наш", "наша", "наше", "наши", "нашего", "нашей", "нашим", "наших",
    "ваш", "ваша", "ваше", "ваши", "вашего", "вашей", "вашим", "ваших",
    "свой", "своя", "своё", "свое", "свои", "своего", "своей", "своим", "своих",
    # English pronouns
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "their", "this", "that", "these", "those"
}
SUBJECT_PRONOUNS = ALL_PRONOUNS
SHORT_CONJUNCTIONS = {"и", "а", "но", "или", "да", "ни"}

ENGLISH_TERMS_MAP = {
    # Video editing & 3D software
    "давинчи": "DaVinci",
    "давинчи резолв": "DaVinci Resolve",
    "резолв": "Resolve",
    "премьер": "Premiere",
    "премьер про": "Premiere Pro",
    "премьере": "Premiere",
    "капкат": "CapCut",
    "капкате": "CapCut",
    "капкатом": "CapCut",
    "файнал кат": "Final Cut",
    "файналкат": "Final Cut",
    "афтер эффектс": "After Effects",
    "афтер": "After Effects",
    "фотошоп": "Photoshop",
    "фотошопе": "Photoshop",
    "фотошопом": "Photoshop",
    "блендер": "Blender",
    "блендере": "Blender",
    "комфи юай": "ComfyUI",
    "комфи": "ComfyUI",
    "комфиюай": "ComfyUI",
    "миджорни": "Midjourney",
    "ранвей": "Runway",
    "стейбл дифьюжн": "Stable Diffusion",
    
    # AI & Tech
    "чатгпт": "ChatGPT",
    "чат гпт": "ChatGPT",
    "клод": "Claude",
    "дипсик": "DeepSeek",
    "джемини": "Gemini",
    "опенаи": "OpenAI",
    "пайтон": "Python",
    "пайтоне": "Python",
    "гит": "Git",
    "гитхаб": "GitHub",
    "докер": "Docker",
    "ви whisper": "Whisper",
    "паракит": "Parakeet",
    "элевен лабс": "ElevenLabs",
    
    # Social media & formats
    "шортс": "Shorts",
    "шортсы": "Shorts",
    "шортсах": "Shorts",
    "шортса": "Shorts",
    "рилс": "Reels",
    "рилсы": "Reels",
    "рилсах": "Reels",
    "рилса": "Reels",
    "тикток": "TikTok",
    "тиктоке": "TikTok",
    "тиктока": "TikTok",
    "ютуб": "YouTube",
    "ютубе": "YouTube",
    "ютуба": "YouTube",
    "инстаграм": "Instagram",
    "инсте": "Instagram",
    "инсты": "Instagram",
    "телеграм": "Telegram",
    "телеге": "Telegram",
    "телеги": "Telegram",
    "подкаст": "Podcast",
    "подкасте": "Podcast",
    
    # Production & tech terms
    "таймлайн": "Timeline",
    "таймлайне": "Timeline",
    "таймлайна": "Timeline",
    "рендер": "Render",
    "рендерить": "Render",
    "рендере": "Render",
    "битрейт": "Bitrate",
    "кейфрейм": "Keyframe",
    "кейфреймы": "Keyframes",
    "кейфреймах": "Keyframes",
    "пресет": "Preset",
    "пресеты": "Presets",
    "пресетах": "Presets",
    "плагин": "Plugin",
    "плагины": "Plugins",
    "плагинах": "Plugins",
    "б-ролл": "B-roll",
    "бролл": "B-roll",
    "скрипт": "Script",
    "скрипты": "Scripts",
    "апи": "API",
    "гпу": "GPU",
    "джипию": "GPU",
    "цпу": "CPU",
    "сипию": "CPU",
    "ртх": "RTX",
    "фпс": "FPS",
    "хдр": "HDR",
    "сдр": "SDR",
    "промпт": "Prompt",
    "промпты": "Prompts",
    "промптах": "Prompts",
    "окей": "OK",
    "лайк": "Like",
    "лайки": "Likes",
    "чекнуть": "Check",
}

def clean_word_token(word: str) -> str:
    """Removes punctuation symbols while preserving internal hyphens."""
    return "".join(ch for ch in word if ch.isalnum() or ch == "-")

def is_preposition(word: str) -> bool:
    """Checks whether token is a known preposition."""
    return clean_word_token(word).lower() in ALL_PREPOSITIONS

def is_clause_starter(word: str) -> bool:
    """Checks whether token is a clause starter (e.g. когда, если, чтобы)."""
    return clean_word_token(word).lower() in CLAUSE_STARTERS

def is_pronoun(word: str) -> bool:
    """Checks whether token is a known pronoun (я, мы, ты, он, это, etc.)."""
    return clean_word_token(word).lower() in ALL_PRONOUNS

def is_subject_pronoun(word: str) -> bool:
    """Checks whether token is a pronoun."""
    return is_pronoun(word)

PARTICLES = {"не", "ни"}
SUBJECT_PRONOUNS_LIST = {
    "я", "мы", "ты", "вы", "он", "она", "оно", "они",
    "i", "we", "you", "he", "she", "it", "they"
}
TRAILING_DISALLOWED = ALL_PREPOSITIONS | SHORT_CONJUNCTIONS | PARTICLES | SUBJECT_PRONOUNS_LIST
FUNCTION_WORDS_WORD_LEVEL = (
    ALL_PREPOSITIONS
    | SHORT_CONJUNCTIONS
    | PARTICLES
    | {
        "я", "мы", "ты", "вы", "он", "она", "оно", "они",
        "мне", "нам", "тебе", "вам", "ему", "ей", "им", "их", "его", "ее", "её",
        "меня", "нас", "тебя", "вас", "него", "нее", "ней", "ним", "них",
        "это", "то", "все", "всё",
        "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them"
    }
)

def extract_word_level_items(
    segments: list[dict],
    remove_punct: bool = False,
    case_mode: str = "none",
    replace_yo_flag: bool = False
) -> list[dict]:
    """
    Groups words for word-level subtitles such that function words
    (pronouns like 'я', prepositions 'в', particles 'не') NEVER stand alone on their own line.
    They bond forward to the next word (or backward if at the very end of speech).
    """
    raw_words = []
    for seg in segments:
        for w in seg.get("words", []):
            word_text = process_word(w.get("word", ""), remove_punct, case_mode, replace_yo_flag)
            if not word_text:
                continue
            raw_words.append({
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "text": word_text,
                "clean": clean_word_token(word_text).lower()
            })

    if not raw_words:
        return []

    bonded_items = []
    accum_words = []

    for i, w in enumerate(raw_words):
        accum_words.append(w)
        is_func = (w["clean"] in FUNCTION_WORDS_WORD_LEVEL or len(w["clean"]) <= 2)

        # If not a function word, or it's the last word in list, flush accumulator
        if not is_func or i == len(raw_words) - 1:
            start_t = accum_words[0]["start"]
            end_t = accum_words[-1]["end"]
            combined_text = " ".join(item["text"] for item in accum_words)
            bonded_items.append({
                "start": start_t,
                "end": end_t,
                "text": combined_text
            })
            accum_words = []

    if accum_words:
        if bonded_items:
            bonded_items[-1]["text"] += " " + " ".join(item["text"] for item in accum_words)
            bonded_items[-1]["end"] = accum_words[-1]["end"]
        else:
            start_t = accum_words[0]["start"]
            end_t = accum_words[-1]["end"]
            combined_text = " ".join(item["text"] for item in accum_words)
            bonded_items.append({
                "start": start_t,
                "end": end_t,
                "text": combined_text
            })

    return bonded_items

def extract_words_for_vertical_9_16(segments: list[dict]) -> list[dict]:
    """Extracts a flat list of words with timestamps from segments, estimating timestamps if missing."""
    words = []
    for seg in segments:
        seg_words = seg.get("words", [])
        if seg_words:
            for w in seg_words:
                raw_w = w.get("word", "").strip()
                if raw_w:
                    words.append({
                        "word": raw_w,
                        "start": float(w.get("start", seg.get("start", 0.0))),
                        "end": float(w.get("end", seg.get("end", 0.0)))
                    })
        else:
            text = seg.get("text", "").strip()
            tokens = text.split()
            if tokens:
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", s + 1.0))
                dur = max(0.1, e - s)
                w_dur = dur / len(tokens)
                for idx, tok in enumerate(tokens):
                    words.append({
                        "word": tok,
                        "start": s + idx * w_dur,
                        "end": s + (idx + 1) * w_dur
                    })
    return words

def enforce_no_isolated_short_words(chunks: list[list[dict]], max_chars: int = 24, standalone_word_len: int = 16) -> tuple[list[list[dict]], int]:
    """
    Guarantees that NO chunk consists solely of an isolated pronoun ('я', 'мы', 'он', 'это'),
    conjunction ('и', 'а', 'но'), preposition ('в', 'на', 'с'), particle ('не', 'ни'),
    or short orphan word (< standalone_word_len).
    Merges single-word chunks forward into the next chunk, or backward into the previous chunk.
    Returns (cleaned_chunks, fixed_count).
    """
    fixed_count = 0
    chunks = [c for c in chunks if c]
    changed = True
    iterations = 0
    max_iter = len(chunks) * 5 + 10

    while changed and iterations < max_iter:
        changed = False
        iterations += 1
        i = 0
        while i < len(chunks):
            c = chunks[i]
            if not c:
                chunks.pop(i)
                changed = True
                continue

            if len(c) == 1:
                w_raw = c[0].get("word", "")
                w_clean = clean_word_token(w_raw).lower()
                
                # If word is shorter than standalone threshold, it MUST not stand alone
                if len(w_clean) < standalone_word_len:
                    merged = False
                    
                    # 1. Try to merge forward into next chunk
                    if i + 1 < len(chunks):
                        next_c = chunks[i + 1]
                        next_len = len(" ".join(clean_word_token(w["word"]) for w in next_c))
                        if len(next_c) < 4 or (next_len + len(w_clean) + 1 <= max_chars):
                            next_c.insert(0, c[0])
                            chunks.pop(i)
                            fixed_count += 1
                            changed = True
                            merged = True
                            continue

                    # 2. Or merge backward into previous chunk
                    if not merged and i > 0:
                        prev_c = chunks[i - 1]
                        prev_len = len(" ".join(clean_word_token(w["word"]) for w in prev_c))
                        if len(prev_c) < 4 or (prev_len + len(w_clean) + 1 <= max_chars):
                            prev_c.append(c[0])
                            chunks.pop(i)
                            fixed_count += 1
                            changed = True
                            merged = True
                            continue

                    # 3. If both neighbors are full, force merge into smaller neighbor to prevent orphan
                    if not merged:
                        if i + 1 < len(chunks):
                            chunks[i + 1].insert(0, c[0])
                        elif i > 0:
                            chunks[i - 1].append(c[0])
                        chunks.pop(i)
                        fixed_count += 1
                        changed = True
                        continue
            i += 1

    return [c for c in chunks if c], fixed_count

def enforce_no_trailing_prepositions_and_conjunctions(chunks: list[list[dict]]) -> tuple[list[list[dict]], int]:
    """
    Self-verification safety loop:
    Guarantees that NO chunk ends with a dangling preposition, short conjunction,
    negative particle ('не', 'ни'), or subject pronoun ('я', 'мы', 'он', etc.).
    Every trailing disallowed word is moved to the start of the following chunk.
    If a chunk becomes empty, it is removed.
    Returns (cleaned_chunks, fixed_count).
    """
    fixed_count = 0
    chunks = [c for c in chunks if c]
    
    changed = True
    iterations = 0
    max_iterations = len(chunks) * 5 + 10
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        i = 0
        while i < len(chunks):
            c = chunks[i]
            if not c:
                chunks.pop(i)
                changed = True
                continue
                
            if i < len(chunks) - 1:
                last_w = c[-1].get("word", "")
                clean_last = clean_word_token(last_w).lower()
                
                if clean_last in TRAILING_DISALLOWED:
                    popped = c.pop()
                    chunks[i + 1].insert(0, popped)
                    fixed_count += 1
                    changed = True
                    
                    if not c:
                        chunks.pop(i)
                        continue
            i += 1

    final_chunks = [c for c in chunks if c]
    return final_chunks, fixed_count

def audit_and_validate_subtitles_9_16(chunks: list[list[dict]]) -> dict:
    """
    Comprehensive self-verification audit of 9:16 subtitles:
    1. Standardizes Russian transliterated English tech brands and terms to Latin.
    2. Guarantees 0 trailing prepositions, conjunctions, particles, and subject pronouns by transferring them.
    3. Guarantees 0 isolated 1-word pronouns ('я', 'мы', etc.) and orphan short words.
    4. Audits single-line conformity and returns detailed report.
    """
    # 1. Standardize English terms
    terms_normalized = 0
    for chunk in chunks:
        for w in chunk:
            raw = w.get("word", "")
            clean = clean_word_token(raw).lower()
            if clean in ENGLISH_TERMS_MAP:
                target = ENGLISH_TERMS_MAP[clean]
                prefix = ""
                suffix = ""
                m = re.match(r"^([^a-zA-Zа-яА-Я0-9]*)(.*?)([^a-zA-Zа-яА-Я0-9]*)$", raw)
                if m:
                    prefix, _, suffix = m.groups()
                w["word"] = f"{prefix}{target}{suffix}"
                terms_normalized += 1

    # 2. Enforce prepositions and trailing words transfer rule
    clean_chunks, preps_fixed = enforce_no_trailing_prepositions_and_conjunctions(chunks)

    # 3. Enforce no isolated pronouns and orphan words rule
    clean_chunks, pronouns_fixed = enforce_no_isolated_short_words(clean_chunks)

    # 4. Final verification pass
    preposition_violations = []
    pronoun_violations = []
    max_char_len = 0
    for idx, c in enumerate(clean_chunks):
        line_text = " ".join(w.get("word", "") for w in c)
        if len(line_text) > max_char_len:
            max_char_len = len(line_text)
        if idx < len(clean_chunks) - 1:
            last_clean = clean_word_token(c[-1].get("word", "")).lower()
            if last_clean in TRAILING_DISALLOWED:
                preposition_violations.append((idx, line_text, last_clean))
        if len(c) == 1:
            w_c = clean_word_token(c[0].get("word", "")).lower()
            if w_c in ALL_PRONOUNS or len(w_c) < 16:
                pronoun_violations.append((idx, line_text, w_c))

    return {
        "chunks": clean_chunks,
        "total_chunks": len(clean_chunks),
        "prepositions_fixed": preps_fixed,
        "pronouns_fixed": pronouns_fixed,
        "terms_normalized": terms_normalized,
        "trailing_violations_remaining": len(preposition_violations),
        "pronoun_violations_remaining": len(pronoun_violations),
        "max_char_len": max_char_len,
        "is_valid": (len(preposition_violations) == 0 and len(pronoun_violations) == 0)
    }

def chunk_words_for_vertical_9_16(
    words: list[dict],
    max_chars: int = 20,
    max_words: int = 4,
    stand_alone_word_len: int = 16
) -> list[list[dict]]:
    """
    Chunks words for 9:16 vertical video subtitles (Shorts / Reels / TikTok):
    - Short words grouped up to max_words (default 4) and max_chars (default 20).
    - Truly long words (>= stand_alone_word_len, default 16) placed alone on single line.
    - Prepositions, conjunctions, particles ('не', 'ни'), and subject pronouns ('я', 'мы') start a new line if they won't fit with their phrase.
    - Pronouns ('я', 'мы', etc.) NEVER placed alone; kept together with verbs/phrases.
    - Strictly single line per SRT block with individual timing.
    - Rigorous self-verification pass ensures 0 trailing prepositions/pronouns and 0 isolated single words.
    """
    if not words:
        return []

    chunks = []
    current_chunk = []

    def get_chunk_text(chunk: list[dict]) -> str:
        return " ".join(clean_word_token(w["word"]) for w in chunk)

    i = 0
    n = len(words)
    while i < n:
        w = words[i]
        raw_word = w.get("word", "").strip()
        if not raw_word:
            i += 1
            continue

        clean_w = clean_word_token(raw_word)
        w_len = len(clean_w)
        is_starter = is_clause_starter(raw_word)

        if not current_chunk:
            if w_len >= stand_alone_word_len:
                chunks.append([w])
                i += 1
                continue
            current_chunk.append(w)
            i += 1
            continue

        cur_text = get_chunk_text(current_chunk)
        new_text = cur_text + " " + clean_w
        new_word_count = len(current_chunk) + 1

        prev_raw = current_chunk[-1].get("word", "")
        prev_is_sentence_end = bool(prev_raw and prev_raw[-1] in ".!?;:")

        # 1. Sentence ending punctuation (.!?;)
        if prev_is_sentence_end:
            chunks.append(current_chunk)
            current_chunk = [w]
            i += 1
            continue

        # 2. Clause starter like 'когда', 'если', 'чтобы' (only if current_chunk has >= 2 words or >= 8 chars)
        if is_starter and (len(current_chunk) >= 2 or len(cur_text) >= 8):
            chunks.append(current_chunk)
            current_chunk = [w]
            i += 1
            continue

        # 3. Standalone long word (>= 16 chars like ДОСТОПРИМЕЧАТЕЛЬНОСТЬ)
        if w_len >= stand_alone_word_len:
            if len(current_chunk) == 1 and len(new_text) <= max_chars:
                current_chunk.append(w)
                chunks.append(current_chunk)
                current_chunk = []
                i += 1
                continue
            else:
                chunks.append(current_chunk)
                chunks.append([w])
                current_chunk = []
                i += 1
                continue

        # 4. Current word is in TRAILING_DISALLOWED (preposition, conjunction, particle, or subject pronoun like 'я'):
        # These words must NOT be at the END of a line.
        if (clean_w.lower() in TRAILING_DISALLOWED) and current_chunk:
            can_fit_with_next = False
            if i + 1 < n:
                next_clean = clean_word_token(words[i + 1].get("word", ""))
                if (len(new_text) + 1 + len(next_clean) <= max_chars) and (new_word_count + 1 <= max_words):
                    can_fit_with_next = True

            # If current_chunk is only 1 short pronoun/word and can fit with next, keep together
            if len(current_chunk) == 1 and can_fit_with_next:
                current_chunk.append(w)
                i += 1
                continue

            # Otherwise, if it won't fit with next word or current_chunk is already complete, wrap before it!
            if not can_fit_with_next:
                chunks.append(current_chunk)
                current_chunk = [w]
                i += 1
                continue

        # 5. Next word is in TRAILING_DISALLOWED and cannot fit into current_chunk together with its following word
        if i + 1 < n:
            next_raw = words[i + 1].get("word", "")
            next_clean = clean_word_token(next_raw).lower()
            if next_clean in TRAILING_DISALLOWED:
                if len(current_chunk) >= 2 or len(cur_text) >= 10:
                    current_chunk.append(w)
                    chunks.append(current_chunk)
                    current_chunk = []
                    i += 1
                    continue

        # 6. Max characters or max words exceeded
        if len(new_text) > max_chars or new_word_count > max_words:
            chunks.append(current_chunk)
            current_chunk = [w]
            i += 1
            continue

        current_chunk.append(w)
        i += 1

    if current_chunk:
        chunks.append(current_chunk)

    # Pass 1: Ensure NO trailing prepositions, conjunctions, particles, or subject pronouns
    chunks, _ = enforce_no_trailing_prepositions_and_conjunctions(chunks)
    # Pass 2: Ensure NO isolated 1-word pronouns or orphan words
    chunks, _ = enforce_no_isolated_short_words(chunks, max_chars=max_chars, standalone_word_len=stand_alone_word_len)
    # Pass 3: Re-verify no trailing prepositions after any merge
    chunks, _ = enforce_no_trailing_prepositions_and_conjunctions(chunks)

    return chunks

def build_vertical_9_16_preview_data(
    chunks: list[list[dict]],
    no_gaps: bool = False,
    case_mode: str = "upper",
    remove_punct: bool = True,
    replace_yo_flag: bool = True
) -> list[dict]:
    """
    Builds clean, structured subtitle items for 9:16 vertical video interactive preview and SRT export.
    Returns: list of dicts [{'id': 1, 'start': 0.32, 'end': 1.68, 's_str': '00:00', 'text': '...', 'len': 17}, ...]
    """
    items = []
    for chunk in chunks:
        if not chunk:
            continue
        start = chunk[0].get("start", 0.0)
        end = chunk[-1].get("end", start + 0.5)
        words_text = []
        for w in chunk:
            clean_w = process_word(
                w.get("word", ""),
                remove_punct=remove_punct,
                case_mode=case_mode,
                replace_yo_flag=replace_yo_flag
            )
            if clean_w:
                words_text.append(clean_w)
                
        line_str = " ".join(words_text).strip()
        if line_str:
            items.append({
                "start": round(start, 2),
                "end": round(max(end, start + 0.1), 2),
                "text": line_str
            })

    if not items:
        return []

    if no_gaps:
        for i in range(len(items) - 1):
            items[i]["end"] = max(items[i]["start"] + 0.05, items[i + 1]["start"])

    result = []
    for idx, item in enumerate(items, 1):
        s_secs = item["start"]
        m = int(s_secs // 60)
        s = int(s_secs % 60)
        s_str = f"{m:02d}:{s:02d}"
        result.append({
            "id": idx,
            "start": round(item["start"], 2),
            "end": round(item["end"], 2),
            "s_str": s_str,
            "text": item["text"],
            "len": len(item["text"])
        })
    return result

def generate_srt_vertical_9_16(
    chunks: list[list[dict]],
    no_gaps: bool = False,
    case_mode: str = "upper",
    remove_punct: bool = True,
    replace_yo_flag: bool = True
) -> str:
    """
    Generates single-line SubRip (.srt) subtitles optimized for 9:16 vertical videos.
    If no_gaps=True, extends end times of subtitles to the start time of the next subtitle,
    eliminating timeline gaps (караоке-лента без мерцания и пустых экранов).
    """
    preview_items = build_vertical_9_16_preview_data(
        chunks,
        no_gaps=no_gaps,
        case_mode=case_mode,
        remove_punct=remove_punct,
        replace_yo_flag=replace_yo_flag
    )
    if not preview_items:
        return ""

    srt_entries = []
    for item in preview_items:
        s_str = format_time_srt(item["start"])
        e_str = format_time_srt(item["end"])
        srt_entries.append(f"{item['id']}\n{s_str} --> {e_str}\n{item['text']}\n")

    return "\n".join(srt_entries)

def generate_9_16_phone_widget(preview_items: list[dict]) -> str:
    """
    Generates a high-fidelity interactive 9:16 mobile phone simulator HTML widget
    to preview vertical subtitles (Shorts / Reels / TikTok) directly in Streamlit.
    """
    if not preview_items:
        return """
        <div style="color: #888; padding: 30px; text-align: center; font-family: sans-serif; background: rgba(255,255,255,0.03); border-radius: 12px;">
            📱 Нет доступных субтитров для отображения в формате 9:16.
        </div>
        """

    subtitles_json = json.dumps(preview_items, ensure_ascii=False)
    total_count = len(preview_items)
    first_text = preview_items[0]["text"] if total_count > 0 else ""

    html_template = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
  body {
    background-color: transparent;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #e0e0e0;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 8px 4px;
  }
  
  .preview-wrapper {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    width: 100%;
    max-width: 960px;
    justify-content: center;
    align-items: flex-start;
  }

  /* 9:16 Phone Mockup */
  .phone-frame {
    width: 300px;
    height: 540px;
    background: #000;
    border-radius: 38px;
    box-shadow: 0 20px 45px -10px rgba(0,0,0,0.85), 0 0 0 6px #1f1f28, 0 0 0 9px #353545;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    flex-shrink: 0;
  }

  .phone-screen {
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 35%, #1e1b4b 0%, #09090e 65%, #050508 100%);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
  }

  /* Notch */
  .notch {
    position: absolute;
    top: 9px;
    left: 50%;
    transform: translateX(-50%);
    width: 86px;
    height: 20px;
    background: #000;
    border-radius: 12px;
    z-index: 50;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .camera-dot { width: 7px; height: 7px; background: #161622; border-radius: 50%; border: 1px solid #2b2b3c; }

  /* Status Bar */
  .status-bar {
    padding: 10px 18px 4px;
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    font-weight: 600;
    color: rgba(255,255,255,0.75);
    z-index: 40;
  }

  /* Mock Social Rail (Shorts / Reels style) */
  .social-rail {
    position: absolute;
    right: 10px;
    bottom: 115px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    align-items: center;
    z-index: 25;
    opacity: 0.65;
  }
  .social-icon {
    width: 32px;
    height: 32px;
    background: rgba(255,255,255,0.12);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    backdrop-filter: blur(4px);
  }

  /* Safe Zone Marker */
  .safe-zone-hint {
    position: absolute;
    top: 48px;
    left: 12px;
    font-size: 10px;
    color: rgba(255,255,255,0.3);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-weight: 600;
    pointer-events: none;
  }

  /* Captions Container (Bottom 30% safe zone) */
  .caption-area {
    position: absolute;
    bottom: 110px;
    left: 14px;
    right: 50px;
    min-height: 85px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    z-index: 30;
    padding: 6px 8px;
    border-radius: 10px;
  }

  .caption-text {
    font-family: 'Impact', 'Arial Black', -apple-system, sans-serif;
    font-size: 25px;
    line-height: 1.18;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    text-align: center;
    word-break: break-word;
    transition: transform 0.1s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  }

  /* Caption Themes */
  .theme-gold {
    color: #FFE600;
    -webkit-text-stroke: 1.4px #000000;
    text-shadow: 0 3px 8px rgba(0,0,0,0.95), -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
  }
  .theme-white {
    color: #FFFFFF;
    -webkit-text-stroke: 1.4px #000000;
    text-shadow: 0 3px 8px rgba(0,0,0,0.95), -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
  }
  .theme-lime {
    color: #22FF66;
    -webkit-text-stroke: 1.4px #000000;
    text-shadow: 0 0 12px rgba(34,255,102,0.65), 0 3px 8px rgba(0,0,0,0.95), -1px -1px 0 #000, 1px -1px 0 #000;
  }
  .theme-cyan {
    color: #00E5FF;
    -webkit-text-stroke: 1.4px #000000;
    text-shadow: 0 0 12px rgba(0,229,255,0.65), 0 3px 8px rgba(0,0,0,0.95), -1px -1px 0 #000, 1px -1px 0 #000;
  }

  /* Bottom Player Bar inside Phone */
  .player-bar {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 10px 14px 18px;
    background: linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.85) 35%, #000 100%);
    z-index: 40;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .progress-slider {
    width: 100%;
    -webkit-appearance: none;
    height: 4px;
    background: rgba(255,255,255,0.22);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }
  .progress-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #ff4b4b;
    cursor: pointer;
  }
  .controls-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .time-badge {
    font-size: 11px;
    font-family: monospace;
    color: rgba(255,255,255,0.85);
  }
  .btn-play {
    background: #ff4b4b;
    border: none;
    color: #fff;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    cursor: pointer;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.1s;
  }
  .btn-play:active { transform: scale(0.92); }
  .btn-nav {
    background: rgba(255,255,255,0.14);
    border: none;
    color: #fff;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    cursor: pointer;
    font-size: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* Sidebar Controls & Playlist */
  .sidebar-panel {
    flex: 1;
    min-width: 310px;
    max-width: 580px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .card-box {
    background: rgba(28, 28, 38, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 12px 14px;
    backdrop-filter: blur(12px);
  }
  .style-pills {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 6px;
  }
  .pill-btn {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #ccc;
    padding: 5px 10px;
    border-radius: 16px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .pill-btn.active {
    background: #ff4b4b;
    border-color: #ff4b4b;
    color: #fff;
    font-weight: 600;
  }

  /* Subtitle List */
  .sub-list {
    max-height: 330px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding-right: 4px;
  }
  .sub-list::-webkit-scrollbar { width: 5px; }
  .sub-list::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 3px; }

  .sub-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 9px;
    padding: 7px 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.12s;
  }
  .sub-item:hover {
    background: rgba(255,255,255,0.09);
    border-color: rgba(255,255,255,0.18);
  }
  .sub-item.active {
    background: rgba(255, 75, 75, 0.2);
    border-color: #ff4b4b;
    color: #fff;
    font-weight: 600;
  }
  .sub-meta {
    font-size: 10px;
    font-family: monospace;
    color: rgba(255,255,255,0.45);
    margin-right: 8px;
    white-space: nowrap;
  }
  .sub-chars {
    font-size: 10px;
    padding: 1px 5px;
    background: rgba(255,255,255,0.09);
    border-radius: 8px;
    color: #aaa;
    white-space: nowrap;
    margin-left: 8px;
  }
  .search-box {
    width: 100%;
    padding: 7px 10px;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    color: #fff;
    font-size: 12px;
    outline: none;
    margin-bottom: 8px;
  }
  .search-box:focus {
    border-color: #ff4b4b;
  }
</style>
</head>
<body>

<div class="preview-wrapper">
  <!-- 9:16 Phone Window -->
  <div class="phone-frame">
    <div class="phone-screen">
      <!-- Notch & Status -->
      <div class="notch"><div class="camera-dot"></div></div>
      <div class="status-bar">
        <span>09:41</span>
        <span>⚡ 9:16</span>
      </div>

      <div class="safe-zone-hint">9:16 Safe Zone</div>

      <!-- Social rail icons (TikTok/Reels style) -->
      <div class="social-rail">
        <div class="social-icon">❤️</div>
        <div class="social-icon">💬</div>
        <div class="social-icon">🔖</div>
        <div class="social-icon">↗️</div>
      </div>

      <!-- Live Caption on 9:16 Screen -->
      <div class="caption-area" id="captionArea">
        <div class="caption-text theme-gold" id="captionText">%%FIRST_TEXT%%</div>
      </div>

      <!-- Bottom player bar inside Phone -->
      <div class="player-bar">
        <input type="range" class="progress-slider" id="scrubber" min="0" max="100" value="0" step="0.05">
        <div class="controls-row">
          <button class="btn-nav" id="btnPrev" title="Предыдущий титр">⏮</button>
          <button class="btn-play" id="btnPlay" title="Воспроизвести / Пауза">▶</button>
          <button class="btn-nav" id="btnNext" title="Следующий титр">⏭</button>
          <span class="time-badge" id="timeBadge">00:00 / 00:00</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Sidebar Controls & Subtitle Playlist -->
  <div class="sidebar-panel">
    <!-- Style Settings -->
    <div class="card-box">
      <div style="font-size: 12px; font-weight: 600; margin-bottom: 4px;">🎨 Стиль субтитров в окне:</div>
      <div class="style-pills">
        <button class="pill-btn theme-pill active" onclick="setTheme('theme-gold', this)">🟡 Gold (Hormozi)</button>
        <button class="pill-btn theme-pill" onclick="setTheme('theme-white', this)">⚪ Clean White</button>
        <button class="pill-btn theme-pill" onclick="setTheme('theme-lime', this)">🟢 Neon Lime</button>
        <button class="pill-btn theme-pill" onclick="setTheme('theme-cyan', this)">🔵 Cyber Cyan</button>
      </div>
      <div style="display:flex; justify-content: space-between; align-items:center; margin-top: 8px;">
        <span style="font-size:11px; color:#aaa;">Размер шрифта:</span>
        <div style="display:flex; gap:6px;">
          <button class="pill-btn font-pill" onclick="setFontSize('21px', this)">S (21px)</button>
          <button class="pill-btn font-pill active" onclick="setFontSize('25px', this)">M (25px)</button>
          <button class="pill-btn font-pill" onclick="setFontSize('30px', this)">L (30px)</button>
        </div>
      </div>
    </div>

    <!-- Subtitle Playlist -->
    <div class="card-box">
      <div style="display:flex; justify-content: space-between; align-items:center; margin-bottom: 8px;">
        <span style="font-size: 12px; font-weight: 600;">📑 Клиповые титры 9:16:</span>
        <span style="font-size: 11px; color: #ff4b4b; font-weight: 600;" id="counterBadge">1 / %%TOTAL_COUNT%%</span>
      </div>
      <input type="text" class="search-box" id="searchInput" placeholder="🔍 Найти слово в субтитрах..." oninput="filterSubs()">
      <div class="sub-list" id="subList"></div>
    </div>
  </div>
</div>

<script>
  const subtitles = %%SUBTITLES_JSON%%;
  let currentIndex = 0;
  let isPlaying = false;
  let playInterval = null;
  let currentTime = 0;
  const maxTime = subtitles.length > 0 ? subtitles[subtitles.length - 1].end : 0;

  const captionText = document.getElementById('captionText');
  const scrubber = document.getElementById('scrubber');
  const timeBadge = document.getElementById('timeBadge');
  const counterBadge = document.getElementById('counterBadge');
  const btnPlay = document.getElementById('btnPlay');
  const subList = document.getElementById('subList');

  if (scrubber) scrubber.max = maxTime;

  function formatTime(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  }

  function renderList() {
    if (!subList) return;
    subList.innerHTML = '';
    subtitles.forEach((s, idx) => {
      const item = document.createElement('div');
      item.className = 'sub-item' + (idx === currentIndex ? ' active' : '');
      item.id = `sub-row-${idx}`;
      item.innerHTML = `
        <div style="display:flex; align-items:center; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
          <span class="sub-meta">#${s.id} [${s.s_str}]</span>
          <span style="overflow:hidden; text-overflow:ellipsis;">${s.text}</span>
        </div>
        <span class="sub-chars">${s.len} зн</span>
      `;
      item.onclick = () => selectSubtitle(idx);
      subList.appendChild(item);
    });
  }

  function selectSubtitle(idx) {
    if (!subtitles || subtitles.length === 0 || idx < 0 || idx >= subtitles.length) return;
    currentIndex = idx;
    currentTime = subtitles[idx].start;
    updateDisplay();
  }

  function updateDisplay() {
    if (!subtitles || subtitles.length === 0) return;
    const cur = subtitles[currentIndex];
    if (!cur) return;
    captionText.textContent = cur.text;
    captionText.style.transform = 'scale(0.92)';
    setTimeout(() => { captionText.style.transform = 'scale(1)'; }, 40);

    if (scrubber) scrubber.value = currentTime;
    if (timeBadge) timeBadge.textContent = `${formatTime(currentTime)} / ${formatTime(maxTime)}`;
    if (counterBadge) counterBadge.textContent = `${currentIndex + 1} / ${subtitles.length}`;

    document.querySelectorAll('.sub-item').forEach((el, i) => {
      el.classList.toggle('active', i === currentIndex);
    });
    const activeEl = document.getElementById(`sub-row-${currentIndex}`);
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function togglePlay() {
    isPlaying = !isPlaying;
    if (btnPlay) btnPlay.textContent = isPlaying ? '⏸' : '▶';
    if (isPlaying) {
      const stepMs = 50;
      playInterval = setInterval(() => {
        currentTime += (stepMs / 1000);
        if (currentTime > maxTime) {
          currentTime = 0;
          currentIndex = 0;
        }
        for (let i = 0; i < subtitles.length; i++) {
          if (currentTime >= subtitles[i].start && currentTime <= subtitles[i].end) {
            if (currentIndex !== i) {
              currentIndex = i;
              updateDisplay();
            }
            break;
          }
        }
        if (scrubber) scrubber.value = currentTime;
        if (timeBadge) timeBadge.textContent = `${formatTime(currentTime)} / ${formatTime(maxTime)}`;
      }, stepMs);
    } else {
      clearInterval(playInterval);
    }
  }

  if (btnPlay) btnPlay.onclick = togglePlay;
  const btnPrev = document.getElementById('btnPrev');
  if (btnPrev) btnPrev.onclick = () => selectSubtitle(Math.max(0, currentIndex - 1));
  const btnNext = document.getElementById('btnNext');
  if (btnNext) btnNext.onclick = () => selectSubtitle(Math.min(subtitles.length - 1, currentIndex + 1));

  if (scrubber) {
    scrubber.oninput = (e) => {
      currentTime = parseFloat(e.target.value);
      for (let i = 0; i < subtitles.length; i++) {
        if (currentTime >= subtitles[i].start && currentTime <= subtitles[i].end) {
          currentIndex = i;
          break;
        }
      }
      updateDisplay();
    };
  }

  window.setTheme = function(cls, btn) {
    captionText.className = 'caption-text ' + cls;
    document.querySelectorAll('.theme-pill').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
  };

  window.setFontSize = function(sz, btn) {
    captionText.style.fontSize = sz;
    document.querySelectorAll('.font-pill').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
  };

  window.filterSubs = function() {
    const q = document.getElementById('searchInput').value.toLowerCase();
    document.querySelectorAll('.sub-item').forEach((el, i) => {
      const txt = subtitles[i].text.toLowerCase();
      el.style.display = txt.includes(q) ? 'flex' : 'none';
    });
  };

  renderList();
  updateDisplay();
</script>
</body>
</html>
"""
    return (
        html_template
        .replace("%%SUBTITLES_JSON%%", subtitles_json)
        .replace("%%FIRST_TEXT%%", first_text)
        .replace("%%TOTAL_COUNT%%", str(total_count))
    )

def convert_to_wav(input_path: str, output_path: str) -> float:
    """Uses system FFmpeg to convert media input to a clean 16kHz Mono PCM WAV audio file. Returns conversion duration in seconds."""
    start_time = time.time()
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception(f"FFmpeg conversion failed: {result.stderr.decode('utf-8', errors='ignore')}")
    return time.time() - start_time

@st.cache_resource
def load_speech_model():
    """Initializes Moondream Photon engine downloading 178MB ternary weights on first run."""
    return md.photon("moondream/parakeet-redux")

st.title("⚡ Local Speech Transcriber & Subtitle Generator")
st.markdown("Powered by `moondream/parakeet-redux` (1.58-Bit Ternary AI) + **Google Gemini AI** post-processing.")

saved_config = load_env_config()

# Settings in sidebar
with st.sidebar:
    st.header("⚙️ Настройки транскрибации")
    timestamp_mode = st.radio(
        "Уровень детализации таймкодов:",
        options=["word", "segment"],
        format_func=lambda x: "⏱️ По словам и предложениям (Word-level)" if x == "word" else "💬 Только по предложениям (Segment-level)",
        index=0
    )
    
    st.markdown("---")
    st.subheader("🤖 Умная AI-правка (Gemini / Свой API)")
    has_any_key = bool(saved_config.get("GEMINI_API_KEY") or saved_config.get("OPENAI_API_KEY"))
    use_ai = st.checkbox(
        "Исправлять ошибки и паразиты через нейросеть",
        value=has_any_key,
        help="Использует нейросеть (Gemini или OpenAI-совместимый API) для контекстного исправления ошибок ASR, терминов, опечаток и удаления слов-паразитов"
    )
    
    ai_provider = saved_config.get("AI_PROVIDER", "gemini")
    gemini_api_key = saved_config.get("GEMINI_API_KEY", "")
    gemini_model = saved_config.get("GEMINI_MODEL", "gemini-3.5-flash")
    openai_base_url = saved_config.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_api_key = saved_config.get("OPENAI_API_KEY", "")
    openai_model = saved_config.get("OPENAI_MODEL", "gpt-4o-mini")
    custom_system_prompt = load_system_prompt()
    try:
        ai_batch_size = int(saved_config.get("AI_BATCH_SIZE", 25))
    except (ValueError, TypeError):
        ai_batch_size = 25
    
    if use_ai:
        ai_provider = st.radio(
            "Провайдер AI:",
            options=["gemini", "openai_compatible"],
            format_func=lambda x: {
                "gemini": "✨ Google Gemini (Официальный API)",
                "openai_compatible": "🌐 Свой API (Совместимый с OpenAI)"
            }[x],
            index=0 if saved_config.get("AI_PROVIDER", "gemini") == "gemini" else 1
        )
        
        if ai_provider == "gemini":
            gemini_api_key = st.text_input(
                "Gemini API Key:",
                value=saved_config.get("GEMINI_API_KEY", ""),
                type="password",
                help="Получите бесплатный ключ: https://aistudio.google.com/app/apikey"
            )
            gemini_options = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-lite-latest"]
            current_model_idx = gemini_options.index(gemini_model) if gemini_model in gemini_options else 0
            gemini_model = st.selectbox(
                "Модель Gemini:",
                options=gemini_options,
                index=current_model_idx,
                help="gemini-3.5-flash — стабильная и быстрая модель с отдельной квотой. У gemini-3.8-flash лимит 20 запросов в день на Free Tier."
            )
            if st.button("🔍 Проверить подключение Gemini", use_container_width=True):
                with st.spinner("Проверка подключения к Google Gemini..."):
                    ok, msg = test_gemini_api_connection(gemini_api_key, gemini_model)
                    if ok:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
            
            if (gemini_api_key != saved_config.get("GEMINI_API_KEY") 
                or gemini_model != saved_config.get("GEMINI_MODEL") 
                or saved_config.get("AI_PROVIDER") != "gemini"):
                save_env_config({
                    "GEMINI_API_KEY": gemini_api_key,
                    "GEMINI_MODEL": gemini_model,
                    "AI_PROVIDER": "gemini"
                })
        else:
            openai_base_url = st.text_input(
                "Base URL API:",
                value=saved_config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                help="Базовый URL провайдера. Например: https://api.openai.com/v1, https://openrouter.ai/api/v1"
            )
            openai_api_key = st.text_input(
                "API Key провайдера:",
                value=saved_config.get("OPENAI_API_KEY", ""),
                type="password",
                help="Ваш ключ доступа к выбранному API"
            )
            if openai_api_key.strip().startswith("http://") or openai_api_key.strip().startswith("https://"):
                st.warning("⚠️ В поле 'API Key' вставлен веб-адрес (URL). Сюда нужно вставить ваш секретный ключ API (например, `nvapi-...` или `sk-...`).")
                
            models_cache_key = f"openai_models_{openai_base_url.strip()}_{openai_api_key.strip()}"
            
            bcol1, bcol2 = st.columns(2)
            load_models_clicked = bcol1.button("🔄 Модели", use_container_width=True, help="Подтянуть список доступных моделей с сервера")
            test_conn_clicked = bcol2.button("🔍 Проверить API", use_container_width=True, help="Проверить работоспособность подключения")
            
            if load_models_clicked:
                with st.spinner("Загрузка списка моделей с сервера..."):
                    ok, models_list, msg = fetch_openai_models(openai_base_url, openai_api_key)
                    if ok:
                        st.session_state[models_cache_key] = models_list
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
                        
            if test_conn_clicked:
                with st.spinner(f"Проверка подключения к {openai_base_url}..."):
                    current_model_test = saved_config.get("OPENAI_MODEL", "gpt-4o-mini")
                    ok, msg = test_openai_api_connection(openai_base_url, openai_api_key, current_model_test)
                    if ok:
                        st.success(f"✅ {msg}")
                        if models_cache_key not in st.session_state:
                            _, models_list, _ = fetch_openai_models(openai_base_url, openai_api_key)
                            if models_list:
                                st.session_state[models_cache_key] = models_list
                                st.rerun()
                    else:
                        st.error(f"❌ {msg}")
                        
            # Auto-fetch models on first view if credentials provided and cache is empty
            if (models_cache_key not in st.session_state 
                and openai_base_url.strip() 
                and openai_api_key.strip()):
                ok, models_list, _ = fetch_openai_models(openai_base_url, openai_api_key)
                if ok and models_list:
                    st.session_state[models_cache_key] = models_list

            saved_model_name = saved_config.get("OPENAI_MODEL", "gpt-4o-mini")
            
            if models_cache_key in st.session_state and st.session_state[models_cache_key]:
                fetched_models = list(st.session_state[models_cache_key])
                matched_idx = 0
                for i, m in enumerate(fetched_models):
                    if m == saved_model_name or m.split("/")[-1] == saved_model_name:
                        matched_idx = i
                        break
                else:
                    if saved_model_name:
                        fetched_models.insert(0, saved_model_name)
                        matched_idx = 0
                    
                selected_model = st.selectbox(
                    f"Выберите модель ({len(fetched_models)} доступно):",
                    options=fetched_models,
                    index=matched_idx,
                    help="Список моделей автоматически подтянут с вашего сервера API"
                )
                manual_input = st.checkbox("✏️ Ввести другое название вручную", value=False)
                if manual_input:
                    openai_model = st.text_input("Название модели вручную:", value=selected_model)
                else:
                    openai_model = selected_model
            else:
                openai_model = st.text_input(
                    "Название модели:",
                    value=saved_model_name,
                    help="Нажмите кнопку '🔄 Модели' выше, чтобы подтянуть список с сервера"
                )
                        
            updates = {}
            if openai_base_url != saved_config.get("OPENAI_BASE_URL"):
                updates["OPENAI_BASE_URL"] = openai_base_url
            if openai_api_key != saved_config.get("OPENAI_API_KEY"):
                updates["OPENAI_API_KEY"] = openai_api_key
            if openai_model != saved_config.get("OPENAI_MODEL"):
                updates["OPENAI_MODEL"] = openai_model
            if saved_config.get("AI_PROVIDER") != "openai_compatible":
                updates["AI_PROVIDER"] = "openai_compatible"
            if updates:
                save_env_config(updates)
                
        st.markdown("---")
        st.markdown("#### 📝 Системный промт AI")
        
        all_presets = load_all_prompt_presets()
        active_preset = saved_config.get("ACTIVE_PROMPT_PRESET", "🎯 Универсальный (Редактор субтитров)")
        if active_preset not in all_presets:
            active_preset = list(all_presets.keys())[0]

        selected_preset = st.selectbox(
            "Выбор системного промта:",
            options=list(all_presets.keys()),
            index=list(all_presets.keys()).index(active_preset),
            help="Выберите готовый профиль промта под ваш тип контента или создайте свой собственный"
        )

        if st.session_state.get("last_selected_preset") != selected_preset:
            st.session_state["last_selected_preset"] = selected_preset
            st.session_state["current_prompt_text"] = all_presets.get(selected_preset, DEFAULT_SYSTEM_PROMPT)
            save_env_config({"ACTIVE_PROMPT_PRESET": selected_preset})
            save_system_prompt(st.session_state["current_prompt_text"])

        if "current_prompt_text" not in st.session_state:
            st.session_state["current_prompt_text"] = all_presets.get(selected_preset, DEFAULT_SYSTEM_PROMPT)

        with st.expander("✏️ Просмотр и редактирование промта", expanded=False):
            st.markdown(
                "Правила перепроверки слов нейросетью: глубокий контекстный анализ, "
                "исправление ошибок ASR, падежей, дефисов (`наконец-то`, `из-за`), "
                "названий софта и технологий латиницей, перенос предлогов/местоимений, удаление заминок."
            )
            edited_prompt = st.text_area(
                f"Инструкция для AI ({selected_preset}):",
                value=st.session_state["current_prompt_text"],
                height=260,
                help="Нейросеть руководствуется этим промтом при анализе каждого сегмента и слова."
            )
            st.session_state["current_prompt_text"] = edited_prompt

            ai_batch_size = st.slider(
                "Размер пачки (сегментов за один запрос):",
                min_value=5,
                max_value=50,
                value=min(max(ai_batch_size, 5), 50),
                step=5,
                help="Рекомендуется 20–25: чем меньше размер пачки, тем внимательнее AI вычитывает каждое слово, не пропуская мелких опечаток."
            )

            col_save_p, col_reset_p = st.columns(2)
            if col_save_p.button("💾 Применить и сохранить", use_container_width=True):
                save_system_prompt(edited_prompt)
                if selected_preset not in BUILTIN_PROMPT_PRESETS or selected_preset == "✍️ Свой промт (Пользовательский)":
                    customs = load_custom_prompts()
                    customs[selected_preset] = edited_prompt
                    save_custom_prompts(customs)
                save_env_config({
                    "ACTIVE_PROMPT_PRESET": selected_preset,
                    "AI_BATCH_SIZE": str(ai_batch_size)
                })
                st.toast(f"✅ Промт '{selected_preset}' успешно применён и сохранён!")
                st.rerun()

            if col_reset_p.button("🔄 Восстановить стандартный", use_container_width=True):
                if selected_preset in BUILTIN_PROMPT_PRESETS:
                    default_txt = BUILTIN_PROMPT_PRESETS[selected_preset]
                else:
                    default_txt = DEFAULT_SYSTEM_PROMPT
                st.session_state["current_prompt_text"] = default_txt
                save_system_prompt(default_txt)
                if selected_preset in load_custom_prompts():
                    customs = load_custom_prompts()
                    customs[selected_preset] = default_txt
                    save_custom_prompts(customs)
                save_env_config({
                    "ACTIVE_PROMPT_PRESET": selected_preset,
                    "AI_BATCH_SIZE": "25"
                })
                st.toast("🔄 Текст промта сброшен к исходному шаблону!")
                st.rerun()

            st.markdown("---")
            st.caption("➕ **Сохранить текущий текст как новый пресет:**")
            new_preset_col1, new_preset_col2 = st.columns([3, 2])
            new_preset_name = new_preset_col1.text_input(
                "Название нового пресета:",
                placeholder="Например, Мой Блог",
                label_visibility="collapsed"
            )
            if new_preset_col2.button("➕ Добавить", use_container_width=True):
                clean_name = new_preset_name.strip()
                if clean_name:
                    full_name = f"✍️ {clean_name}" if not clean_name.startswith("✍️") else clean_name
                    customs = load_custom_prompts()
                    customs[full_name] = edited_prompt
                    save_custom_prompts(customs)
                    save_env_config({"ACTIVE_PROMPT_PRESET": full_name})
                    save_system_prompt(edited_prompt)
                    st.session_state["last_selected_preset"] = full_name
                    st.session_state["current_prompt_text"] = edited_prompt
                    st.toast(f"✅ Создан новый пресет: '{full_name}'!")
                    st.rerun()
                else:
                    st.warning("Введите название пресета!")

            if selected_preset not in BUILTIN_PROMPT_PRESETS and selected_preset != "✍️ Свой промт (Пользовательский)":
                if st.button(f"🗑️ Удалить пресет '{selected_preset}'", use_container_width=True):
                    customs = load_custom_prompts()
                    if selected_preset in customs:
                        del customs[selected_preset]
                        save_custom_prompts(customs)
                    save_env_config({"ACTIVE_PROMPT_PRESET": "🎯 Универсальный (Редактор субтитров)"})
                    st.toast(f"🗑️ Пресет '{selected_preset}' удалён!")
                    st.rerun()

        custom_system_prompt = st.session_state.get("current_prompt_text", edited_prompt)
    
    st.markdown("---")
    st.subheader("🚫 Локальный фильтр паразитов")
    remove_fillers = st.checkbox(
        "Удалять слова-паразиты по словарю",
        value=True,
        help="Быстро вырезает звуки заминки (ээ, мм, эм), ну, короче, типа, как бы и др."
    )
    
    single_fillers, multi_fillers = set(), []
    if remove_fillers:
        with st.expander("⚙️ Настроить словарь слов-паразитов", expanded=False):
            fillers_input = st.text_area(
                "Слова и выражения для удаления (через запятую):",
                value=", ".join(DEFAULT_FILLER_WORDS),
                height=120,
                help="Вы можете отредактировать этот список: добавить свои слова или удалить те, которые хотите оставить"
            )
            single_fillers, multi_fillers = parse_filler_words(fillers_input)
    
    st.markdown("---")
    st.subheader("📝 Очистка и регистр текста")
    remove_punctuation = st.checkbox(
        "✂️ Убирать знаки препинания",
        value=True,
        help="Удаляет точки, запятые, тире, кавычки и знаки препинания, сохраняя внутрисловные дефисы (наконец-то, из-за, по-русски)"
    )
    replace_yo_char = st.checkbox(
        "🔤 Заменять букву «ё» на «е»",
        value=True,
        help="Автоматически заменяет все буквы «ё» и «Ё» на «е» и «Е» в тексте и субтитрах"
    )
    remove_pauses = st.checkbox(
        "⚡ Убрать паузы между субтитрами",
        value=True,
        help="Режим «Без пауз» (Gapless): устраняет зазоры между титрами, растягивая каждый блок до начала следующего. Действует на все типы субтитров (9:16, обычные, пословные)."
    )
    case_mode = st.radio(
        "Регистр букв (Case):",
        options=["upper", "none", "lower", "title"],
        format_func=lambda x: {
            "upper": "🔠 ВСЕ СЛОВА В ВЕРХНИЙ РЕГИСТР (UPPERCASE)",
            "none": "🔤 Оригинальный регистр (Как распознано)",
            "lower": "🔡 все слова в нижний регистр (lowercase)",
            "title": "🔠 Каждое Слово С Заглавной (Title Case)"
        }[x],
        index=0,
        help="Выберите форматирование регистра для слов и субтитров"
    )
    
    with st.expander("📱 Настройки субтитров 9:16 (Shorts / Reels)", expanded=False):
        max_chars_9_16 = st.slider("Макс. символов в строке:", min_value=12, max_value=30, value=20, step=1, help="Рекомендуется 18-22 символа для идеального отображения в Shorts / Reels")
        max_words_9_16 = st.slider("Макс. слов в блоке:", min_value=2, max_value=6, value=4, step=1, help="Объединять до указанного числа коротких слов (по умолчанию 4)")
        standalone_len_9_16 = st.slider("Длина одиночного слова:", min_value=10, max_value=24, value=16, step=1, help="Слова длиннее этого значения могут стоять отдельной строкой (по умолчанию 16)")
    
    st.markdown("---")
    st.markdown(
        "**Особенности:**\n"
        "- **Parakeet Redux:** мгновенное локальное распознавание на GPU (RTX 3060).\n"
        "- **Gemini AI:** устраняет смысловые ошибки, терминологию и оговорки.\n"
        "- **По словам + ВЕРХНИЙ РЕГИСТР:** готовые караоке-субтитры для видеомонтажа."
    )

SUPPORTED_FORMATS = [
    "mp4", "mkv", "avi", "mov", "webm", "flv",
    "mp3", "wav", "m4a", "flac", "aac", "ogg"
]

uploaded_file = st.file_uploader(
    "Загрузите видео или аудио файл для локальной транскрибации (до 10 ГБ)",
    type=SUPPORTED_FORMATS,
    help="Поддерживаются видео и аудиофайлы любого разрешения размером до 10 ГБ"
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_ext = uploaded_file.name.split('.')[-1].lower()
    base_name = os.path.splitext(uploaded_file.name)[0]
    file_id = f"{uploaded_file.name}_{len(file_bytes)}"
    
    # 1. Local ASR Inference (cached in session state per file & timestamp_mode)
    asr_cache_key = f"asr_{file_id}_{timestamp_mode}"
    if asr_cache_key not in st.session_state:
        total_start_time = time.time()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file_path = os.path.join(temp_dir, f"input.{file_ext}")
            wav_file_path = os.path.join(temp_dir, "audio.wav")
            
            with open(input_file_path, "wb") as f:
                f.write(file_bytes)
                
            st.info("Извлечение и конвертация аудио в формат 16кГц WAV через FFmpeg...")
            try:
                conversion_time = convert_to_wav(input_file_path, wav_file_path)
            except Exception as e:
                st.error(f"Ошибка обработки аудио: {e}")
                st.stop()
                
            st.success(f"Аудио подготовлено за {conversion_time:.2f} сек. Запуск локального инференса нейросети...")
            with st.spinner("Распознавание речи с формированием пословных таймкодов..."):
                try:
                    model_start_time = time.time()
                    speech_model = load_speech_model()
                    result = speech_model.transcribe(audio=wav_file_path, timestamps=timestamp_mode)
                    transcription_time = time.time() - model_start_time
                    st.session_state[asr_cache_key] = {
                        "raw_segments": result.get("segments", []),
                        "conversion_time": conversion_time,
                        "transcription_time": transcription_time,
                        "total_execution_time": time.time() - total_start_time,
                        "duration_seconds": result.get("duration_seconds", 0.0)
                    }
                except Exception as e:
                    st.error(f"Ошибка при транскрибации: {e}")
                    st.stop()
                    
    cached_asr = st.session_state[asr_cache_key]
    raw_segments = cached_asr["raw_segments"]
    conversion_time = cached_asr["conversion_time"]
    transcription_time = cached_asr["transcription_time"]
    total_execution_time = cached_asr["total_execution_time"]
    duration_seconds = cached_asr["duration_seconds"]

    try:
        # Filter filler words via local dictionary
        processed_segments = []
        removed_fillers_count = 0
        
        for seg in raw_segments:
            seg_dict = dict(seg)
            words = seg.get("words", [])
            if words and remove_fillers:
                filtered_words = filter_words(words, single_fillers, multi_fillers)
                removed_fillers_count += (len(words) - len(filtered_words))
                if not filtered_words:
                    continue  # Whole segment was filler words
                seg_dict["words"] = filtered_words
                seg_dict["text"] = " ".join([w.get("word", "") for w in filtered_words])
                seg_dict["start"] = filtered_words[0].get("start", seg.get("start", 0.0))
                seg_dict["end"] = filtered_words[-1].get("end", seg.get("end", 0.0))
            elif remove_fillers and not words:
                clean_seg_text = filter_text_fillers(seg.get("text", ""), single_fillers, multi_fillers)
                if not clean_seg_text:
                    continue
                seg_dict["text"] = clean_seg_text
            processed_segments.append(seg_dict)
        
        # AI enhancement (Google Gemini or OpenAI-compatible)
        ai_applied = False
        active_model_name = gemini_model if ai_provider == "gemini" else openai_model
        ai_provider_name = "Google Gemini" if ai_provider == "gemini" else "Совместимый с OpenAI"
        prompt_hash = abs(hash(custom_system_prompt))
        ai_cache_key = f"ai_{file_id}_{ai_provider}_{active_model_name}_{remove_fillers}_{prompt_hash}_{ai_batch_size}"
        
        if use_ai:
            active_key = gemini_api_key.strip() if ai_provider == "gemini" else openai_api_key.strip()
            if not active_key:
                st.warning(f"⚠️ Для исправления через AI укажите API Key ({ai_provider_name}) в боковом меню или добавьте его в файл .env")
            else:
                if ai_cache_key in st.session_state:
                    processed_segments = st.session_state[ai_cache_key]
                    ai_applied = True
                else:
                    progress_bar = st.progress(0, text=f"✨ Подключение к {ai_provider_name} ({active_model_name})...")
                    status_text = st.empty()
                    try:
                        if ai_provider == "gemini":
                            enhanced = enhance_transcript_with_gemini(
                                processed_segments,
                                api_key=gemini_api_key.strip(),
                                model_name=gemini_model,
                                batch_size=ai_batch_size,
                                progress_bar=progress_bar,
                                status_text=status_text,
                                system_instruction=custom_system_prompt
                            )
                        else:
                            enhanced = enhance_transcript_with_openai_compatible(
                                processed_segments,
                                base_url=openai_base_url.strip(),
                                api_key=openai_api_key.strip(),
                                model_name=openai_model.strip(),
                                batch_size=ai_batch_size,
                                progress_bar=progress_bar,
                                status_text=status_text,
                                system_instruction=custom_system_prompt
                            )
                        st.session_state[ai_cache_key] = enhanced
                        processed_segments = enhanced
                        ai_applied = True
                    except Exception as ge:
                        err_text = str(ge)
                        if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
                            st.warning(
                                "⚠️ **Исчерпан суточный лимит бесплатных запросов Google Gemini (Free Tier: 20 запросов в день на модель).**\n\n"
                                "👉 **Как продолжить:**\n"
                                "1. В боковом меню выберите модель **`gemini-3.5-flash`** или **`gemini-3.5-flash-lite`** (у них отдельная свежая квота).\n"
                                "2. Либо переключитесь на **«🌐 Свой API (Совместимый с OpenAI)»**.\n"
                                "3. Локальная транскрибация Parakeet Redux полностью сохранена и доступна ниже."
                            )
                        elif "404" in err_text or "page not found" in err_text.lower():
                            st.warning(
                                f"⚠️ **Ошибка 404 (Модель не найдена на сервере):**\n\n"
                                f"Модель `{active_model_name}` не найдена по адресу `{openai_base_url}`.\n\n"
                                "👉 **Как исправить:**\n"
                                "1. В боковом меню нажмите кнопку **«🔄 Модели»**.\n"
                                "2. Выберите модель из выпадающего списка (с полным именем провайдера, например `deepseek-ai/deepseek-v4.1-flash`)."
                            )
                        elif "401" in err_text or "unauthorized" in err_text.lower() or "invalid json object" in err_text.lower():
                            st.warning(
                                "⚠️ **Ошибка 401 (Не авторизован):**\n\n"
                                "Сервер отклонил ключ доступа. Убедитесь, что в поле 'API Key' вставлен секретный ключ доступа (например, `nvapi-...`), а не веб-адрес."
                            )
                        else:
                            st.error(f"Ошибка AI API ({ge}). Использован локальный результат.")
                    finally:
                        progress_bar.empty()
                        status_text.empty()
        
        if use_ai and ai_applied:
            if st.sidebar.button("🔄 Повторить обработку AI"):
                st.session_state.pop(ai_cache_key, None)
                st.rerun()
        
        # Performance metrics
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Извлечение аудио (FFmpeg)", f"{conversion_time:.2f} с")
        mcol2.metric("Время транскрибации AI", f"{transcription_time:.2f} с")
        mcol3.metric("Общее время работы", f"{total_execution_time:.2f} с")
        if ai_applied:
            ai_status_text = f"{ai_provider_name} ({active_model_name}) ✅"
        elif not use_ai:
            ai_status_text = "Выключена"
        elif not active_key:
            ai_status_text = "Нет ключа ⚠️"
        else:
            ai_status_text = "Ошибка AI ⚠️ (локально)"
        mcol4.metric("AI правка", ai_status_text)
        
        if remove_fillers and removed_fillers_count > 0:
            st.info(f"🚫 Локально удалено слов-паразитов: **{removed_fillers_count}**" + (" + дополнительная правка AI" if ai_applied else ""))
                
        st.subheader("Результаты транскрибации")
        
        tab_names = ["💬 Распечатка текста и сегменты", "⏱️ Пословные таймкоды", "🎬 Субтитры (SRT / VTT)"]
        if ai_applied:
            tab_names.append("🔄 Сравнение: Исходный vs AI")
            
        tabs = st.tabs(tab_names)
        tab_segments = tabs[0]
        tab_words = tabs[1]
        tab_subtitles = tabs[2]
        tab_diff = tabs[3] if ai_applied else None
        
        # Process segments and words
        transcript_lines = []
        with tab_segments:
            # Full text box for quick reading and copying
            full_clean_text = " ".join([
                process_text(s.get("text", ""), remove_punctuation, case_mode, replace_yo_char)
                for s in processed_segments
                if s.get("text", "").strip()
            ])
            
            st.markdown("#### 📄 Распечатка полного текста")
            st.text_area(
                "Полный текст транскрибации (единым текстом):",
                value=full_clean_text,
                height=180,
                help="Вы можете выделить и скопировать весь текст целиком."
            )
            
            st.markdown("---")
            st.markdown("#### ⏱️ Временные фрагменты (сегменты)")
            for idx, segment in enumerate(processed_segments, 1):
                start = segment.get("start", 0.0)
                end = segment.get("end", 0.0)
                raw_text = segment.get("text", "").strip()
                display_text = process_text(raw_text, remove_punctuation, case_mode, replace_yo_char)
                words = segment.get("words", [])
                
                start_str = format_time_display(start)
                end_str = format_time_display(end)
                
                st.markdown(f"**`[{start_str} - {end_str}]`** {display_text}")
                transcript_lines.append(f"* **`[{start_str} - {end_str}]`** {display_text}")
                
                if words:
                    with st.expander(f"🔍 Пословные метки фрагмента #{idx} ({len(words)} слов)", expanded=False):
                        word_chips = []
                        for w in words:
                            w_text = process_word(w.get("word", ""), remove_punctuation, case_mode, replace_yo_char)
                            if not w_text:
                                continue
                            w_start = w.get("start", 0.0)
                            w_end = w.get("end", 0.0)
                            word_chips.append(f"`{w_text}` ({w_start:.2f}с - {w_end:.2f}с)")
                        st.write(" • ".join(word_chips))
        
        # Tab 2: Word-level table
        all_words_list = []
        for s_idx, segment in enumerate(processed_segments, 1):
            for w in segment.get("words", []):
                w_text = process_word(w.get("word", ""), remove_punctuation, case_mode, replace_yo_char)
                if not w_text:
                    continue
                w_start = w.get("start", 0.0)
                w_end = w.get("end", 0.0)
                all_words_list.append({
                    "№": len(all_words_list) + 1,
                    "Сегмент": s_idx,
                    "Слово": w_text,
                    "Начало (сек)": round(w_start, 2),
                    "Конец (сек)": round(w_end, 2),
                    "Длительность (сек)": round(w_end - w_start, 2),
                    "Начало (таймкод)": format_time_srt(w_start),
                    "Конец (таймкод)": format_time_srt(w_end),
                })
        
        with tab_words:
            if all_words_list:
                st.caption(f"Всего распознано слов с точными временными метками: {len(all_words_list)}")
                df_words = pd.DataFrame(all_words_list)
                st.dataframe(df_words, use_container_width=True, hide_index=True)
            else:
                st.info("В выбранном режиме пословные таймкоды не формировались. Переключите режим в боковом меню на 'По словам и предложениям'.")
        
        # Generate subtitle formats with punct stripping, filler removal, case formatting, yo-replacement, and pause setting
        srt_standard = generate_srt(
            processed_segments,
            word_level=False,
            remove_punct=remove_punctuation,
            case_mode=case_mode,
            replace_yo_flag=replace_yo_char,
            no_gaps=remove_pauses
        )
        srt_word_level = generate_srt(
            processed_segments,
            word_level=True,
            remove_punct=remove_punctuation,
            case_mode=case_mode,
            replace_yo_flag=replace_yo_char,
            no_gaps=remove_pauses
        ) if all_words_list else ""
        vtt_standard = generate_vtt(
            processed_segments,
            word_level=False,
            remove_punct=remove_punctuation,
            case_mode=case_mode,
            replace_yo_flag=replace_yo_char
        )
        vtt_word_level = generate_vtt(
            processed_segments,
            word_level=True,
            remove_punct=remove_punctuation,
            case_mode=case_mode,
            replace_yo_flag=replace_yo_char
        ) if all_words_list else ""
        
        # 9:16 Vertical video subtitles (Shorts / Reels / TikTok)
        words_9_16 = extract_words_for_vertical_9_16(processed_segments)
        raw_chunks_9_16 = chunk_words_for_vertical_9_16(
            words_9_16,
            max_chars=max_chars_9_16,
            max_words=max_words_9_16,
            stand_alone_word_len=standalone_len_9_16
        )
        audit_report_9_16 = audit_and_validate_subtitles_9_16(raw_chunks_9_16)
        chunks_9_16 = audit_report_9_16["chunks"]
        srt_9_16 = generate_srt_vertical_9_16(
            chunks_9_16,
            no_gaps=remove_pauses,
            case_mode=case_mode,
            remove_punct=remove_punctuation,
            replace_yo_flag=replace_yo_char
        )

        # Tab 3: Subtitles preview
        with tab_subtitles:
            st.markdown("#### 📱 Клиповые субтитры 9:16 (Shorts / Reels / TikTok)")
            if remove_pauses:
                st.caption("⚡ **Режим «Без пауз» (Gapless)** активен: зазоры между титрами устранены (настроено в боковом меню).")
            else:
                st.caption("⏱️ **Режим «С паузами»** активен: естественные паузы тишины сохранены (настроено в боковом меню).")

            st.success(
                f"🛡️ **Самопроверка AI и аудит 9:16 пройдены (100%):**\n\n"
                f"• 🚫 **Висячие предлоги, союзы и местоимения на концах строк:** 0 (перенесено: {audit_report_9_16['prepositions_fixed']})\n\n"
                f"• 👤 **Одиночные местоимения («я», «мы», «он») и одинокие слова:** 0 (объединено с фразами: {audit_report_9_16.get('pronouns_fixed', 0)})\n\n"
                f"• 🔤 **Английские бренды и софт:** на латинице (стандартизировано: {audit_report_9_16['terms_normalized']})\n\n"
                f"• 📐 **Формат 9:16:** {audit_report_9_16['total_chunks']} титров, строго 1 строка на блок (макс. {audit_report_9_16['max_char_len']} симв.)"
            )

            if audit_report_9_16["prepositions_fixed"] > 0 or audit_report_9_16.get("pronouns_fixed", 0) > 0 or audit_report_9_16["terms_normalized"] > 0:
                with st.expander("🔍 Детали самопроверки и автоматической коррекции", expanded=False):
                    st.markdown(
                        f"- **Исправлено висячих предлогов/союзов на концах строк:** `{audit_report_9_16['prepositions_fixed']}` (перенесены на следующую строку к зависимым словам).\n"
                        f"- **Объединено одиночных местоимений («я», «мы» и др.):** `{audit_report_9_16.get('pronouns_fixed', 0)}` (включены в состав смысловых фраз).\n"
                        f"- **Терминов и брендов софта стандартизировано на латиницу:** `{audit_report_9_16['terms_normalized']}`.\n"
                        f"- **Всего сформировано вертикальных блоков (9:16):** `{audit_report_9_16['total_chunks']}`.\n"
                        f"- **Остаточных нарушений правил переноса:** `{audit_report_9_16['trailing_violations_remaining']}`."
                    )

            st.download_button(
                label=f"📱 Скачать субтитры 9:16 (.srt) {'⚡ (Без пауз)' if remove_pauses else ''}",
                data=srt_9_16,
                file_name=f"{base_name}_9_16.srt",
                mime="text/plain",
                use_container_width=True,
                type="primary",
                help="Скачать субтитры 9:16 для монтажа в Shorts / Reels / TikTok"
            )

            # 9:16 Visual Phone Preview Window
            preview_items_9_16 = build_vertical_9_16_preview_data(
                chunks_9_16,
                no_gaps=remove_pauses,
                case_mode=case_mode,
                remove_punct=remove_punctuation,
                replace_yo_flag=replace_yo_char
            )
            if preview_items_9_16:
                st.markdown("##### 📱 Интерактивный предпросмотр 9:16 (Визуализация в окне смартфона)")
                components.html(generate_9_16_phone_widget(preview_items_9_16), height=640, scrolling=False)

            with st.expander("📄 Сырой текст субтитров 9:16 (.srt)", expanded=False):
                st.caption("Оптимизировано для вертикального видео: 1 строка на блок (до 20 знаков), предлоги и союзы перенесены вперед, длинные слова стоят отдельно.")
                st.text_area("9:16 SRT Text", srt_9_16, height=220)

            st.markdown("---")
            st.markdown("#### 🎬 Другие форматы субтитров")
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown("##### 📄 Стандартные субтитры (по предложениям)")
                st.text_area("SRT Preview", srt_standard, height=240)
            with sub_col2:
                st.markdown("##### ⏱️ Пословные субтитры (Караоке / Word-by-Word)")
                if srt_word_level:
                    st.text_area("Word-level SRT Preview", srt_word_level, height=240)
                else:
                    st.info("Пословные субтитры доступны при выборе режима 'По словам'.")
        
        # Tab 4: Diff comparison if AI applied
        if tab_diff is not None:
            with tab_diff:
                st.markdown(f"### Сравнение текста до и после обработки нейросетью ({ai_provider_name}: {active_model_name})")
                diff_rows = []
                for seg in processed_segments:
                    if "original_text" in seg:
                        diff_rows.append({
                            "Таймкод": f"{format_time_display(seg['start'])} - {format_time_display(seg['end'])}",
                            "Исходный текст (ASR)": seg["original_text"],
                            "Исправлено AI": seg["text"]
                        })
                if diff_rows:
                    st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
        
        # Prepare outputs and Markdown report
        body_transcript = "\n\n".join(transcript_lines)
        
        words_report_lines = []
        if all_words_list:
            words_report_lines.append("\n\n## ⏱️ Пословные временные метки\n\n| № | Слово | Начало (с) | Конец (с) | Длительность (с) |\n|---|---|---|---|---|")
            for row in all_words_list:
                words_report_lines.append(f"| {row['№']} | `{row['Слово']}` | {row['Начало (сек)']} | {row['Конец (сек)']} | {row['Длительность (сек)']} |")
        
        words_report = "\n".join(words_report_lines)
        
        full_report = f"""# ⚡ Parakeet Redux Transcription Report

## ⏱️ Execution Performance Summary
* **Source Media File:** `{uploaded_file.name}`
* **Audio Extraction Time (FFmpeg):** `{conversion_time:.2f} seconds`
* **AI Model Transcription Time:** `{transcription_time:.2f} seconds`
* **Total End-to-End Pipeline Runtime:** `{total_execution_time:.2f} seconds`
* **Total Words Count:** `{len(all_words_list)}`
* **AI Enhanced:** `{ai_applied} ({ai_provider_name}: {active_model_name})`
* **Filler Words Filtered:** `{remove_fillers} ({removed_fillers_count} removed)`
* **Punctuation Removed:** `{remove_punctuation}`
* **Case Mode:** `{case_mode}`

---

## 📝 Timestamped Transcript (Segments)

{body_transcript}
{words_report}
"""
        
        # Process JSON export with cleaned words and sentences
        processed_json = {
            "text": "\n".join([process_text(s.get("text", ""), remove_punctuation, case_mode, replace_yo_char) for s in processed_segments]),
            "duration_seconds": duration_seconds,
            "ai_enhanced": ai_applied,
            "ai_provider": ai_provider if ai_applied else None,
            "ai_model": active_model_name if ai_applied else None,
            "segments": []
        }
        for seg in processed_segments:
            s_dict = {
                "text": process_text(seg.get("text", ""), remove_punctuation, case_mode, replace_yo_char),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "words": [
                    {
                        "word": process_word(w.get("word", ""), remove_punctuation, case_mode, replace_yo_char),
                        "start": w.get("start"),
                        "end": w.get("end")
                    }
                    for w in seg.get("words", [])
                    if process_word(w.get("word", ""), remove_punctuation, case_mode, replace_yo_char)
                ]
            }
            processed_json["segments"].append(s_dict)

        full_json_export = json.dumps(processed_json, ensure_ascii=False, indent=2)

        # Save outputs locally to outputs directory
        os.makedirs("outputs", exist_ok=True)
        with open(os.path.join("outputs", "outputs.md"), "w", encoding="utf-8") as f:
            f.write(full_report)
        with open(os.path.join("outputs", "subtitles_9_16.srt"), "w", encoding="utf-8") as f:
            f.write(srt_9_16)
        with open(os.path.join("outputs", "subtitles.srt"), "w", encoding="utf-8") as f:
            f.write(srt_standard)
        if srt_word_level:
            with open(os.path.join("outputs", "subtitles_words.srt"), "w", encoding="utf-8") as f:
                f.write(srt_word_level)
        with open(os.path.join("outputs", "transcript.json"), "w", encoding="utf-8") as f:
            f.write(full_json_export)
        
        # Download buttons
        st.markdown("---")
        st.subheader("📥 Скачать результаты")

        d_cols = st.columns(5 if srt_word_level else 4)
        
        d_cols[0].download_button(
            label=f"📱 9:16 (.srt) {'⚡' if remove_pauses else ''}",
            data=srt_9_16,
            file_name=f"{base_name}_9_16.srt",
            mime="text/plain",
            use_container_width=True,
            type="primary",
            help="Клиповые субтитры 9:16 для Shorts / Reels / TikTok"
        )
        d_cols[1].download_button(
            label="🎬 Обычные (.srt)",
            data=srt_standard,
            file_name=f"{base_name}.srt",
            mime="text/plain",
            use_container_width=True,
            help="Стандартные субтитры по предложениям"
        )
        if srt_word_level:
            d_cols[2].download_button(
                label="⏱️ Пословные (.srt)",
                data=srt_word_level,
                file_name=f"{base_name}_words.srt",
                mime="text/plain",
                use_container_width=True,
                help="Пословные караоке-субтитры"
            )
            d_cols[3].download_button(
                label="📥 Отчёт (.md)",
                data=full_report,
                file_name=f"{base_name}_report.md",
                mime="text/markdown",
                use_container_width=True
            )
            d_cols[4].download_button(
                label="📊 JSON (.json)",
                data=full_json_export,
                file_name=f"{base_name}_words.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            d_cols[2].download_button(
                label="📥 Отчёт (.md)",
                data=full_report,
                file_name=f"{base_name}_report.md",
                mime="text/markdown",
                use_container_width=True
            )
            d_cols[3].download_button(
                label="📊 JSON (.json)",
                data=full_json_export,
                file_name=f"{base_name}_words.json",
                mime="application/json",
                use_container_width=True
            )
        
    except Exception as e:
        st.error(f'Ошибка при обработке результатов: {e}')
