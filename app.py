import streamlit as st
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

def load_env_config() -> dict[str, str]:
    """Loads key-value configuration from local .env file and environment variables."""
    config = {
        "AI_PROVIDER": os.environ.get("AI_PROVIDER", "gemini"),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "GEMINI_MODEL": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "OPENAI_MODEL": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
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
    batch_size: int = 50,
    progress_bar = None,
    status_text = None
) -> list[dict]:
    """Uses Google Gemini API to fix transcription errors, remove speech fillers, and polish words while preserving and aligning timestamps."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip())
    
    system_instruction = (
        "Ты — профессиональный редактор субтитров для видеомонтажа. "
        "Твоя цель — сделать так, чтобы субтитры точно соответствовали звучащей речи, но были грамотными, чистыми и без мусора:\n"
        "1. Исправь ошибки распознавания речи (ASR): фонетические искажения, опечатки, неправильно услышанные технические термины и названия "
        "(Claude, ChatGPT, Codex, Kimi, DaVinci, Premiere, нейросеть, localhost, Git и т.д.), падежи и окончания.\n"
        "2. Полностью удали слова-паразиты, звуки заминки речи (э-э, мм, ну, типа, короче, как бы, в общем, так сказать и т.д.), заикания и случайные повторы слов.\n"
        "3. ВАЖНО: НЕ ПЕРЕСКАЗЫВАЙ и НЕ ПЕРЕФРАЗИРУЙ речь! Сохраняй авторские формулировки и структуру предложений, чтобы зритель слышал ровно то, что написано в субтитрах.\n"
        "4. Сохрани количество и порядок переданных сегментов. Верни строго валидный JSON-массив объектов: [{\"id\": 0, \"text\": \"...\"}]. "
        "Если фрагмент состоял исключительно из заминки или паразитов, верни text: \"\"."
    )
    
    all_corrections = {}
    total_segments = len(segments)
    
    # Process in batches to guarantee fast, reliable responses without timeout or token overflow
    for start_idx in range(0, total_segments, batch_size):
        end_idx = min(start_idx + batch_size, total_segments)
        batch = segments[start_idx:end_idx]
        
        items_to_send = [{"id": start_idx + j, "text": seg.get("text", "")} for j, seg in enumerate(batch)]
        prompt = (
            f"Вот сегменты речи #{start_idx + 1}–#{end_idx} (всего {len(items_to_send)}) в формате JSON:\n"
            f"{json.dumps(items_to_send, ensure_ascii=False, indent=2)}\n\n"
            "Исправь ошибки и удали слова-паразиты. Сохраняй авторские слова, не перефразируй! "
            "Верни строго валидный JSON-массив объектов с полями id и text."
        )
        
        if status_text is not None:
            status_text.text(f"✨ Gemini ({model_name}): обработка сегментов {start_idx + 1}–{end_idx} из {total_segments}...")
            
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
                        system_instruction=system_instruction,
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
    batch_size: int = 50,
    progress_bar = None,
    status_text = None
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
    
    system_instruction = (
        "Ты — профессиональный редактор субтитров для видеомонтажа. "
        "Твоя цель — сделать так, чтобы субтитры точно соответствовали звучащей речи, но были грамотными, чистыми и без мусора:\n"
        "1. Исправь ошибки распознавания речи (ASR): фонетические искажения, опечатки, неправильно услышанные технические термины и названия "
        "(Claude, ChatGPT, Codex, Kimi, DaVinci, Premiere, нейросеть, localhost, Git и т.д.), падежи и окончания.\n"
        "2. Полностью удали слова-паразиты, звуки заминки речи (э-э, мм, ну, типа, короче, как бы, в общем, так сказать и т.д.), заикания и случайные повторы слов.\n"
        "3. ВАЖНО: НЕ ПЕРЕСКАЗЫВАЙ и НЕ ПЕРЕФРАЗИРУЙ речь! Сохраняй авторские формулировки и структуру предложений, чтобы зритель слышал ровно то, что написано в субтитрах.\n"
        "4. Сохрани количество и порядок переданных сегментов. Верни строго валидный JSON-массив объектов: [{\"id\": 0, \"text\": \"...\"}]. "
        "Если фрагмент состоял исключительно из заминки или паразитов, верни text: \"\"."
    )
    
    all_corrections = {}
    total_segments = len(segments)
    
    for start_idx in range(0, total_segments, batch_size):
        end_idx = min(start_idx + batch_size, total_segments)
        batch = segments[start_idx:end_idx]
        
        items_to_send = [{"id": start_idx + j, "text": seg.get("text", "")} for j, seg in enumerate(batch)]
        prompt = (
            f"Вот сегменты речи #{start_idx + 1}–#{end_idx} (всего {len(items_to_send)}) в формате JSON:\n"
            f"{json.dumps(items_to_send, ensure_ascii=False, indent=2)}\n\n"
            "Исправь ошибки и удали слова-паразиты. Сохраняй авторские слова, не перефразируй! "
            "Верни строго валидный JSON-массив объектов с полями id и text."
        )
        
        if status_text is not None:
            status_text.text(f"✨ AI ({resolved_model}): обработка сегментов {start_idx + 1}–{end_idx} из {total_segments}...")
            
        raw_text = ""
        try:
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw_text = resp.choices[0].message.content or ""
        except Exception:
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_instruction},
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
    items = []
    if word_level:
        for seg in segments:
            for w in seg.get("words", []):
                word_text = process_word(w.get("word", ""), remove_punct, case_mode, replace_yo_flag)
                if not word_text:
                    continue
                items.append({
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "text": word_text
                })
    else:
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
    idx = 1
    if word_level:
        for seg in segments:
            for w in seg.get("words", []):
                word_text = process_word(w.get("word", ""), remove_punct, case_mode, replace_yo_flag)
                if not word_text:
                    continue
                start = format_time_vtt(w.get("start", 0.0))
                end = format_time_vtt(w.get("end", 0.0))
                lines.append(f"{idx}\n{start} --> {end}\n{word_text}\n")
                idx += 1
    else:
        for seg in segments:
            text = process_text(seg.get("text", ""), remove_punct, case_mode, replace_yo_flag)
            if not text:
                continue
            start = format_time_vtt(seg.get("start", 0.0))
            end = format_time_vtt(seg.get("end", 0.0))
            lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
            idx += 1
    return "\n".join(lines)

RUSSIAN_PREPOSITIONS = {
    "в", "во", "с", "со", "к", "ко", "о", "об", "обо", "у", "на", "по", "от", "ото",
    "до", "из", "изо", "за", "под", "подо", "над", "надо", "при", "про", "без", "безо",
    "для", "через", "сквозь", "перед", "передо", "ради", "около", "возле", "среди",
    "вокруг", "между", "из-за", "из-под"
}

ENGLISH_PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
    "into", "through", "under", "over", "before", "after", "between"
}

ALL_PREPOSITIONS = RUSSIAN_PREPOSITIONS | ENGLISH_PREPOSITIONS

CLAUSE_STARTERS = {
    "когда", "если", "что", "чтобы", "потому", "как", "где", "куда", "откуда",
    "хотя", "зачем", "почему", "который", "которая", "которое", "которые",
    "when", "if", "that", "because", "how", "where", "why", "which"
}

SUBJECT_PRONOUNS = {"я", "мы", "ты", "вы", "он", "она", "оно", "они"}
SHORT_CONJUNCTIONS = {"и", "а", "но", "или", "да"}

def clean_word_token(word: str) -> str:
    """Removes punctuation symbols while preserving internal hyphens."""
    return "".join(ch for ch in word if ch.isalnum() or ch == "-")

def is_preposition(word: str) -> bool:
    """Checks whether token is a known preposition."""
    return clean_word_token(word).lower() in ALL_PREPOSITIONS

def is_clause_starter(word: str) -> bool:
    """Checks whether token is a clause starter (e.g. когда, если, чтобы)."""
    return clean_word_token(word).lower() in CLAUSE_STARTERS

def is_subject_pronoun(word: str) -> bool:
    """Checks whether token is a personal subject pronoun (я, мы, ты, etc.)."""
    return clean_word_token(word).lower() in SUBJECT_PRONOUNS

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

def chunk_words_for_vertical_9_16(
    words: list[dict],
    max_chars: int = 20,
    max_words: int = 4,
    stand_alone_word_len: int = 12
) -> list[list[dict]]:
    """
    Chunks words for 9:16 vertical video subtitles (Shorts / Reels / TikTok):
    - Short words grouped up to max_words (default 4) and max_chars (default 20).
    - Long words (>= stand_alone_word_len, default 12) placed alone on single line.
    - Prepositions always start a new line with their phrase (never dangling at line end).
    - Clause starters ('когда', 'если') and subject pronouns ('мы', 'я') start a new chunk.
    - Strictly single line per SRT block with individual timing.
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
        is_prep = is_preposition(raw_word)
        is_starter = is_clause_starter(raw_word)
        is_pron = is_subject_pronoun(raw_word)

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
        prev_clean = clean_word_token(prev_raw).lower()
        prev_is_sentence_end = bool(prev_raw and prev_raw[-1] in ".!?;:")

        # 1. Sentence ending punctuation (.!?;)
        if prev_is_sentence_end:
            chunks.append(current_chunk)
            current_chunk = [w]
            i += 1
            continue

        # 2. Clause starter like 'когда', 'если', 'чтобы'
        if is_starter:
            chunks.append(current_chunk)
            current_chunk = [w]
            i += 1
            continue

        # 3. Subject pronoun ('мы', 'я', etc.) starting a new clause after a noun/verb
        if is_pron and (prev_raw.endswith(",") or (len(current_chunk) >= 1 and prev_clean not in ALL_PREPOSITIONS and prev_clean not in CLAUSE_STARTERS)):
            chunks.append(current_chunk)
            current_chunk = [w]
            i += 1
            continue

        # 4. Standalone long word (>= 12 chars like ПРИВЕТСТВОВАТЬ)
        if w_len >= stand_alone_word_len:
            if len(current_chunk) == 1 and is_preposition(current_chunk[0].get("word", "")) and len(new_text) <= max_chars:
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

        # 5. Current word is a preposition
        if is_prep:
            if len(current_chunk) >= 2 or len(cur_text) >= 10:
                chunks.append(current_chunk)
                current_chunk = [w]
                i += 1
                continue
            if i + 1 < n:
                next_clean = clean_word_token(words[i + 1].get("word", ""))
                if len(cur_text) + 1 + len(clean_w) + 1 + len(next_clean) > max_chars:
                    chunks.append(current_chunk)
                    current_chunk = [w]
                    i += 1
                    continue

        # 6. Next word is a preposition and would cause overflow or current is already complete
        if i + 1 < n and is_preposition(words[i + 1].get("word", "")):
            if len(current_chunk) >= 2 or len(cur_text) >= 10:
                chunks.append(current_chunk)
                current_chunk = [w]
                i += 1
                continue

        # 7. Max characters or max words exceeded
        if len(new_text) > max_chars or new_word_count > max_words:
            chunks.append(current_chunk)
            current_chunk = [w]
            i += 1
            continue

        current_chunk.append(w)
        i += 1

    if current_chunk:
        chunks.append(current_chunk)

    # Safety pass: Ensure NO chunk ends with a dangling preposition or short conjunction
    final_chunks = []
    for idx, c in enumerate(chunks):
        if len(c) > 1 and (is_preposition(c[-1].get("word", "")) or clean_word_token(c[-1].get("word", "")).lower() in SHORT_CONJUNCTIONS):
            dangling = c.pop()
            if idx + 1 < len(chunks):
                chunks[idx + 1].insert(0, dangling)
            else:
                c.append(dangling)
        if c:
            final_chunks.append(c)

    return final_chunks

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
                "start": start,
                "end": max(end, start + 0.1),
                "text": line_str
            })

    if not items:
        return ""

    if no_gaps:
        for i in range(len(items) - 1):
            items[i]["end"] = max(items[i]["start"] + 0.05, items[i + 1]["start"])

    srt_entries = []
    for idx, item in enumerate(items, 1):
        s_str = format_time_srt(item["start"])
        e_str = format_time_srt(item["end"])
        srt_entries.append(f"{idx}\n{s_str} --> {e_str}\n{item['text']}\n")

    return "\n".join(srt_entries)

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
    remove_pauses_default = st.checkbox(
        "⚡ Убрать паузы между субтитрами",
        value=True,
        help="Режим «Без пауз» (Gapless): устраняет зазоры между титрами, растягивая каждый блок до начала следующего"
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
        standalone_len_9_16 = st.slider("Длина одиночного слова:", min_value=8, max_value=16, value=12, step=1, help="Слова длиннее этого значения будут стоять отдельной строкой (по умолчанию 12)")
    
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
    "Загрузите видео или аудио файл для локальной транскрибации",
    type=SUPPORTED_FORMATS
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
        ai_cache_key = f"ai_{file_id}_{ai_provider}_{active_model_name}_{remove_fillers}"
        
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
                                batch_size=50,
                                progress_bar=progress_bar,
                                status_text=status_text
                            )
                        else:
                            enhanced = enhance_transcript_with_openai_compatible(
                                processed_segments,
                                base_url=openai_base_url.strip(),
                                api_key=openai_api_key.strip(),
                                model_name=openai_model.strip(),
                                batch_size=50,
                                progress_bar=progress_bar,
                                status_text=status_text
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
        
        # Generate subtitle formats with punct stripping, filler removal, case formatting, and yo-replacement
        srt_standard = generate_srt(processed_segments, word_level=False, remove_punct=remove_punctuation, case_mode=case_mode, replace_yo_flag=replace_yo_char, no_gaps=False)
        srt_standard_gapless = generate_srt(processed_segments, word_level=False, remove_punct=remove_punctuation, case_mode=case_mode, replace_yo_flag=replace_yo_char, no_gaps=True)
        srt_word_level = generate_srt(processed_segments, word_level=True, remove_punct=remove_punctuation, case_mode=case_mode, replace_yo_flag=replace_yo_char, no_gaps=False) if all_words_list else ""
        srt_word_level_gapless = generate_srt(processed_segments, word_level=True, remove_punct=remove_punctuation, case_mode=case_mode, replace_yo_flag=replace_yo_char, no_gaps=True) if all_words_list else ""
        vtt_standard = generate_vtt(processed_segments, word_level=False, remove_punct=remove_punctuation, case_mode=case_mode, replace_yo_flag=replace_yo_char)
        vtt_word_level = generate_vtt(processed_segments, word_level=True, remove_punct=remove_punctuation, case_mode=case_mode, replace_yo_flag=replace_yo_char) if all_words_list else ""
        
        # 9:16 Vertical video subtitles (Shorts / Reels / TikTok)
        words_9_16 = extract_words_for_vertical_9_16(processed_segments)
        chunks_9_16 = chunk_words_for_vertical_9_16(
            words_9_16,
            max_chars=max_chars_9_16,
            max_words=max_words_9_16,
            stand_alone_word_len=standalone_len_9_16
        )
        srt_9_16_gapless = generate_srt_vertical_9_16(
            chunks_9_16,
            no_gaps=True,
            case_mode=case_mode,
            remove_punct=remove_punctuation,
            replace_yo_flag=replace_yo_char
        )
        srt_9_16 = generate_srt_vertical_9_16(
            chunks_9_16,
            no_gaps=False,
            case_mode=case_mode,
            remove_punct=remove_punctuation,
            replace_yo_flag=replace_yo_char
        )

        # Tab 3: Subtitles preview
        with tab_subtitles:
            top_h1, top_h2 = st.columns([1, 1])
            with top_h1:
                st.markdown("#### 📱 Клиповые субтитры 9:16 (Shorts / Reels / TikTok)")
            with top_h2:
                remove_gaps_toggle = st.toggle(
                    "⚡ Убрать паузы (Режим без пауз / Gapless)",
                    value=remove_pauses_default,
                    help="Устраняет любые зазоры между титрами: титры идут непрерывным потоком без черных кадров"
                )

            if remove_gaps_toggle:
                st.success("⚡ Режим «УБРАТЬ ПАУЗЫ» активен: конец каждого титра состыкован со стартом следующего без зазоров.")
            else:
                st.info("⏱️ Режим «С паузами»: сохранены естественные паузы тишины между репликами.")

            # Dedicated prominent button right in Tab 3
            btn_t1, btn_t2 = st.columns(2)
            with btn_t1:
                st.download_button(
                    label="⚡ КНОПКА: УБРАТЬ ПАУЗЫ (Скачать 9:16 .srt)",
                    data=srt_9_16_gapless,
                    file_name=f"{base_name}_9_16_no_pauses.srt",
                    mime="text/plain",
                    use_container_width=True,
                    type="primary",
                    help="Скачать субтитры 9:16 без зазоров между фразами для моментального монтажа"
                )
            with btn_t2:
                st.download_button(
                    label="📱 Скачать 9:16 с паузами (.srt)",
                    data=srt_9_16,
                    file_name=f"{base_name}_9_16_with_pauses.srt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Субтитры 9:16 с сохранением естественных пауз"
                )

            st.caption("Оптимизировано для вертикального видео: 1 строка на блок (до 20 знаков), предлоги и союзы перенесены вперед, длинные слова стоят отдельно.")
            sub_916_col1, sub_916_col2 = st.columns(2)
            with sub_916_col1:
                st.markdown("##### ⚡ 9:16 Без пауз (Рекомендуется для Reels/Shorts)")
                st.caption("Каждый блок продлен до старта следующего (без темного экрана и мигания).")
                st.text_area("9:16 Gapless SRT Preview", srt_9_16_gapless, height=260)
            with sub_916_col2:
                st.markdown("##### ⏱️ 9:16 С естественными паузами")
                st.caption("Таймкоды сохраняют зазоры и естественные паузы между фразами.")
                st.text_area("9:16 Standard SRT Preview", srt_9_16, height=260)

            st.markdown("---")
            st.markdown("#### 🎬 Другие форматы субтитров")
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown("##### 📄 Стандартные субтитры (по предложениям)")
                st.text_area("SRT Preview", srt_standard_gapless if remove_gaps_toggle else srt_standard, height=240)
            with sub_col2:
                st.markdown("##### ⏱️ Пословные субтитры (Караоке / Word-by-Word)")
                if srt_word_level:
                    st.text_area("Word-level SRT Preview", srt_word_level_gapless if remove_gaps_toggle else srt_word_level, height=240)
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
        with open(os.path.join("outputs", "subtitles_9_16_no_pauses.srt"), "w", encoding="utf-8") as f:
            f.write(srt_9_16_gapless)
        with open(os.path.join("outputs", "subtitles_9_16_gapless.srt"), "w", encoding="utf-8") as f:
            f.write(srt_9_16_gapless)
        with open(os.path.join("outputs", "subtitles_9_16.srt"), "w", encoding="utf-8") as f:
            f.write(srt_9_16)
        with open(os.path.join("outputs", "subtitles_no_pauses.srt"), "w", encoding="utf-8") as f:
            f.write(srt_standard_gapless)
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

        st.markdown("##### 📱 Клиповые субтитры 9:16 (Shorts / Reels / TikTok)")
        v_col1, v_col2 = st.columns(2)
        v_col1.download_button(
            label="⚡ КНОПКА: УБРАТЬ ПАУЗЫ (9:16 .srt)",
            data=srt_9_16_gapless,
            file_name=f"{base_name}_9_16_no_pauses.srt",
            mime="text/plain",
            use_container_width=True,
            type="primary",
            help="Рекомендуется для Reels/Shorts: таймкоды без разрывов, субтитры не моргают"
        )
        v_col2.download_button(
            label="📱 9:16 С паузами (.srt)",
            data=srt_9_16,
            file_name=f"{base_name}_9_16_with_pauses.srt",
            mime="text/plain",
            use_container_width=True,
            help="Субтитры 9:16 с сохранением естественных пауз тишины"
        )

        st.markdown("##### 🎬 Другие форматы и отчёты")
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        
        dcol1.download_button(
            label="🎬 Обычные (.srt)",
            data=srt_standard_gapless if remove_gaps_toggle else srt_standard,
            file_name=f"{base_name}.srt",
            mime="text/plain",
            use_container_width=True
        )
        if srt_word_level:
            dcol2.download_button(
                label="⏱️ Пословный SRT (.srt)",
                data=srt_word_level_gapless if remove_gaps_toggle else srt_word_level,
                file_name=f"{base_name}_words.srt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            dcol2.empty()
        
        dcol3.download_button(
            label="📥 Отчёт (.md)",
            data=full_report,
            file_name=f"{base_name}_report.md",
            mime="text/markdown",
            use_container_width=True
        )
        dcol4.download_button(
            label="📊 Данные JSON (.json)",
            data=full_json_export,
            file_name=f"{base_name}_words.json",
            mime="application/json",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f'Ошибка при обработке результатов: {e}')
