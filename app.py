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
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "https://api.reformboss.com/v1"),
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
        client.models.generate_content(
            model=model_name.strip() if model_name.strip() else "gemini-3.8-flash",
            contents="ping"
        )
        latency = time.time() - start_t
        return True, f"Успешное подключение к Google Gemini! Время отклика: {latency:.2f}с."
    except Exception as e:
        return False, f"Ошибка подключения к Gemini: {e}"

def test_openai_api_connection(base_url: str, api_key: str, model_name: str) -> tuple[bool, str]:
    """Tests connection to an OpenAI-compatible API endpoint (e.g. ReformBoss, OpenRouter, vLLM)."""
    import time
    from openai import OpenAI
    
    clean_url = base_url.strip().rstrip("/")
    if not clean_url:
        return False, "Укажите Base URL провайдера (например, https://api.reformboss.com/v1)"
    if not api_key.strip():
        return False, "Укажите API Key"
        
    start_t = time.time()
    target_model = model_name.strip() if model_name.strip() else "gpt-4o-mini"
    try:
        client = OpenAI(base_url=clean_url, api_key=api_key.strip(), timeout=12.0)
        res = client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=2
        )
        latency = time.time() - start_t
        reply = res.choices[0].message.content if res.choices else "OK"
        return True, f"Подключение к {clean_url} успешно! Модель '{target_model}' ответила за {latency:.2f}с: '{reply}'"
    except Exception as e:
        try:
            client = OpenAI(base_url=clean_url, api_key=api_key.strip(), timeout=8.0)
            models = client.models.list()
            latency = time.time() - start_t
            model_ids = [m.id for m in models.data[:5]] if hasattr(models, "data") else []
            return True, f"Подключение к {clean_url} успешно (время: {latency:.2f}с). Доступные модели: {', '.join(model_ids)}"
        except Exception:
            return False, f"Ошибка подключения: {e}"

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

def process_text(text: str, remove_punct: bool = False, case_mode: str = "none") -> str:
    """Applies text formatting options (punctuation removal, case formatting)."""
    res = text
    if remove_punct:
        res = strip_punctuation_from_text(res)
    if case_mode == "upper":
        res = res.upper()
    elif case_mode == "lower":
        res = res.lower()
    elif case_mode == "title":
        res = res.title()
    return res

def process_word(word: str, remove_punct: bool = False, case_mode: str = "none") -> str:
    """Applies word token formatting options."""
    res = word
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
    model_name: str = "gemini-3.8-flash",
    batch_size: int = 25,
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
            
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )
        )
        
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
    status_text = None
) -> list[dict]:
    """Uses an OpenAI-compatible API (ReformBoss, OpenRouter, vLLM, DeepSeek, Local LLM)
    to fix transcription errors, remove speech fillers, and polish words while preserving and aligning timestamps."""
    from openai import OpenAI
    
    clean_base_url = base_url.strip().rstrip("/")
    client = OpenAI(base_url=clean_base_url, api_key=api_key.strip(), timeout=90.0)
    
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
            status_text.text(f"✨ AI ({model_name}): обработка сегментов {start_idx + 1}–{end_idx} из {total_segments}...")
            
        raw_text = ""
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw_text = resp.choices[0].message.content or ""
        except Exception:
            resp = client.chat.completions.create(
                model=model_name,
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

def generate_srt(segments: list[dict], word_level: bool = False, remove_punct: bool = False, case_mode: str = "none") -> str:
    """Generates SubRip (.srt) subtitle content from segments or word-level timing."""
    lines = []
    idx = 1
    if word_level:
        for seg in segments:
            for w in seg.get("words", []):
                word_text = process_word(w.get("word", ""), remove_punct, case_mode)
                if not word_text:
                    continue
                start = format_time_srt(w.get("start", 0.0))
                end = format_time_srt(w.get("end", 0.0))
                lines.append(f"{idx}\n{start} --> {end}\n{word_text}\n")
                idx += 1
    else:
        for seg in segments:
            text = process_text(seg.get("text", ""), remove_punct, case_mode)
            if not text:
                continue
            start = format_time_srt(seg.get("start", 0.0))
            end = format_time_srt(seg.get("end", 0.0))
            lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
            idx += 1
    return "\n".join(lines)

def generate_vtt(segments: list[dict], word_level: bool = False, remove_punct: bool = False, case_mode: str = "none") -> str:
    """Generates WebVTT (.vtt) subtitle content from segments or word-level timing."""
    lines = ["WEBVTT\n"]
    idx = 1
    if word_level:
        for seg in segments:
            for w in seg.get("words", []):
                word_text = process_word(w.get("word", ""), remove_punct, case_mode)
                if not word_text:
                    continue
                start = format_time_vtt(w.get("start", 0.0))
                end = format_time_vtt(w.get("end", 0.0))
                lines.append(f"{idx}\n{start} --> {end}\n{word_text}\n")
                idx += 1
    else:
        for seg in segments:
            text = process_text(seg.get("text", ""), remove_punct, case_mode)
            if not text:
                continue
            start = format_time_vtt(seg.get("start", 0.0))
            end = format_time_vtt(seg.get("end", 0.0))
            lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
            idx += 1
    return "\n".join(lines)

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
    gemini_model = "gemini-3.8-flash"
    openai_base_url = saved_config.get("OPENAI_BASE_URL", "https://api.reformboss.com/v1")
    openai_api_key = saved_config.get("OPENAI_API_KEY", "")
    openai_model = saved_config.get("OPENAI_MODEL", "gpt-4o-mini")
    
    if use_ai:
        ai_provider = st.radio(
            "Провайдер AI:",
            options=["gemini", "openai_compatible"],
            format_func=lambda x: {
                "gemini": "✨ Google Gemini (Официальный API)",
                "openai_compatible": "🌐 Свой API (OpenAI-совместимый: ReformBoss, Proxy, Local)"
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
            gemini_model = st.selectbox(
                "Модель Gemini:",
                options=["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
                index=0,
                help="gemini-3.8-flash — актуальная сверхбыстрая и качественная модель"
            )
            if st.button("🔍 Проверить подключение Gemini", use_container_width=True):
                with st.spinner("Проверка подключения к Google Gemini..."):
                    ok, msg = test_gemini_api_connection(gemini_api_key, gemini_model)
                    if ok:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
            
            if gemini_api_key != saved_config.get("GEMINI_API_KEY") or saved_config.get("AI_PROVIDER") != "gemini":
                save_env_config({"GEMINI_API_KEY": gemini_api_key, "AI_PROVIDER": "gemini"})
        else:
            openai_base_url = st.text_input(
                "Base URL API:",
                value=saved_config.get("OPENAI_BASE_URL", "https://api.reformboss.com/v1"),
                help="Базовый URL провайдера. Например: https://api.reformboss.com/v1, https://api.openai.com/v1, https://openrouter.ai/api/v1"
            )
            openai_api_key = st.text_input(
                "API Key провайдера:",
                value=saved_config.get("OPENAI_API_KEY", ""),
                type="password",
                help="Ваш ключ доступа к выбранному API"
            )
            openai_model = st.text_input(
                "Название модели:",
                value=saved_config.get("OPENAI_MODEL", "gpt-4o-mini"),
                help="Например: gpt-4o-mini, gpt-4o, claude-3-5-sonnet, deepseek-chat"
            )
            if st.button("🔍 Проверить подключение к API", use_container_width=True):
                with st.spinner(f"Проверка подключения к {openai_base_url}..."):
                    ok, msg = test_openai_api_connection(openai_base_url, openai_api_key, openai_model)
                    if ok:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
                        
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
        ai_provider_name = "Google Gemini" if ai_provider == "gemini" else "OpenAI API"
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
                                batch_size=25,
                                progress_bar=progress_bar,
                                status_text=status_text
                            )
                        else:
                            enhanced = enhance_transcript_with_openai_compatible(
                                processed_segments,
                                base_url=openai_base_url.strip(),
                                api_key=openai_api_key.strip(),
                                model_name=openai_model.strip(),
                                batch_size=25,
                                progress_bar=progress_bar,
                                status_text=status_text
                            )
                        st.session_state[ai_cache_key] = enhanced
                        processed_segments = enhanced
                        ai_applied = True
                    except Exception as ge:
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
        ai_status_text = f"{ai_provider_name} ({active_model_name}) ✅" if ai_applied else ("Нет ключа ⚠️" if use_ai else "Выключена")
        mcol4.metric("AI правка", ai_status_text)
        
        if remove_fillers and removed_fillers_count > 0:
            st.info(f"🚫 Локально удалено слов-паразитов: **{removed_fillers_count}**" + (" + дополнительная правка AI" if ai_applied else ""))
                
        st.subheader("Результаты транскрибации")
        
        tab_names = ["💬 Предложения и сегменты", "⏱️ Пословные таймкоды", "🎬 Субтитры (SRT / VTT)"]
        if ai_applied:
            tab_names.append("🔄 Сравнение: Исходный vs AI")
            
        tabs = st.tabs(tab_names)
        tab_segments = tabs[0]
        tab_words = tabs[1]
        tab_subtitles = tabs[2]
        tab_diff = tabs[3] if gemini_applied else None
        
        # Process segments and words
        transcript_lines = []
        with tab_segments:
            for idx, segment in enumerate(processed_segments, 1):
                start = segment.get("start", 0.0)
                end = segment.get("end", 0.0)
                raw_text = segment.get("text", "").strip()
                display_text = process_text(raw_text, remove_punctuation, case_mode)
                words = segment.get("words", [])
                
                start_str = format_time_display(start)
                end_str = format_time_display(end)
                
                st.markdown(f"**`[{start_str} - {end_str}]`** {display_text}")
                transcript_lines.append(f"* **`[{start_str} - {end_str}]`** {display_text}")
                
                if words:
                    with st.expander(f"🔍 Пословные метки фрагмента #{idx} ({len(words)} слов)", expanded=False):
                        word_chips = []
                        for w in words:
                            w_text = process_word(w.get("word", ""), remove_punctuation, case_mode)
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
                w_text = process_word(w.get("word", ""), remove_punctuation, case_mode)
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
        
        # Generate subtitle formats with punct stripping, filler removal, and case formatting
        srt_standard = generate_srt(processed_segments, word_level=False, remove_punct=remove_punctuation, case_mode=case_mode)
        srt_word_level = generate_srt(processed_segments, word_level=True, remove_punct=remove_punctuation, case_mode=case_mode) if all_words_list else ""
        vtt_standard = generate_vtt(processed_segments, word_level=False, remove_punct=remove_punctuation, case_mode=case_mode)
        vtt_word_level = generate_vtt(processed_segments, word_level=True, remove_punct=remove_punctuation, case_mode=case_mode) if all_words_list else ""
        
        # Tab 3: Subtitles preview
        with tab_subtitles:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown("#### 📄 Стандартные субтитры (по предложениям)")
                st.text_area("SRT Preview", srt_standard, height=300)
            with sub_col2:
                st.markdown("#### ⏱️ Пословные субтитры (Караоке / Word-by-Word)")
                if srt_word_level:
                    st.text_area("Word-level SRT Preview", srt_word_level, height=300)
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
            "text": "\n".join([process_text(s.get("text", ""), remove_punctuation, case_mode) for s in processed_segments]),
            "duration_seconds": duration_seconds,
            "ai_enhanced": ai_applied,
            "ai_provider": ai_provider if ai_applied else None,
            "ai_model": active_model_name if ai_applied else None,
            "segments": []
        }
        for seg in processed_segments:
            s_dict = {
                "text": process_text(seg.get("text", ""), remove_punctuation, case_mode),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "words": [
                    {
                        "word": process_word(w.get("word", ""), remove_punctuation, case_mode),
                        "start": w.get("start"),
                        "end": w.get("end")
                    }
                    for w in seg.get("words", [])
                    if process_word(w.get("word", ""), remove_punctuation, case_mode)
                ]
            }
            processed_json["segments"].append(s_dict)

        full_json_export = json.dumps(processed_json, ensure_ascii=False, indent=2)

        # Save outputs locally to outputs directory
        os.makedirs("outputs", exist_ok=True)
        with open(os.path.join("outputs", "outputs.md"), "w", encoding="utf-8") as f:
            f.write(full_report)
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
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        
        dcol1.download_button(
            label="📥 Отчёт (.md)",
            data=full_report,
            file_name=f"{base_name}_report.md",
            mime="text/markdown"
        )
        dcol2.download_button(
            label="🎬 Субтитры (.srt)",
            data=srt_standard,
            file_name=f"{base_name}.srt",
            mime="text/plain"
        )
        if srt_word_level:
            dcol3.download_button(
                label="⏱️ Пословный SRT (.srt)",
                data=srt_word_level,
                file_name=f"{base_name}_words.srt",
                mime="text/plain"
            )
        else:
            dcol3.empty()
        
        dcol4.download_button(
            label="📊 Данные JSON (.json)",
            data=full_json_export,
            file_name=f"{base_name}_words.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f'Ошибка при обработке результатов: {e}')
