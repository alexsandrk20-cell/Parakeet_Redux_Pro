import streamlit as st
import tempfile
import os
import time
import subprocess
import json
import re
import string
import pandas as pd
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

def load_saved_api_key() -> str:
    """Loads saved Gemini API Key from environment or .env file."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        return line.strip().split("=", 1)[1].strip("\"' ")
        except Exception:
            pass
    return ""

def save_api_key(key: str) -> None:
    """Saves Gemini API Key to local .env file."""
    if not key:
        return
    try:
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write(f"GEMINI_API_KEY={key.strip()}\n")
    except Exception:
        pass

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

def enhance_transcript_with_gemini(
    segments: list[dict],
    api_key: str,
    model_name: str = "gemini-3.8-flash"
) -> list[dict]:
    """Uses Google Gemini API to fix transcription errors, remove speech fillers, and polish sentences while preserving timestamps."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip())
    
    items_to_send = [{"id": i, "text": seg.get("text", "")} for i, seg in enumerate(segments)]
    
    system_instruction = (
        "Ты — профессиональный редактор стенограмм, субтитров и транскрипций речи. "
        "Твоя задача — исправить распознанный текст речи:\n"
        "1. Исправь ошибки автоматического распознавания (фонетические искажения, опечатки, неправильно понятые термины, профессиональный сленг).\n"
        "2. Полностью удали все слова-паразиты, звуки заминки речи (э-э, мм, ну, типа, короче, как бы, в общем, так сказать и т.д.), повторы и оговорки.\n"
        "3. Сохрани точный смысл каждого предложения и естественность языка.\n"
        "4. Верни JSON-массив объектов с теми же id и исправленным полем text: [{\"id\": 0, \"text\": \"...\"}]. "
        "Если фрагмент состоял исключительно из заминки, верни text: \"\"."
    )
    
    prompt = f"Вот распознанные сегменты речи в формате JSON:\n{json.dumps(items_to_send, ensure_ascii=False, indent=2)}\n\nИсправь ошибки и удали слова-паразиты. Верни строго валидный JSON-массив объектов с id и text."
    
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
    corrections_map = {item["id"]: item.get("text", "").strip() for item in corrected_items if "id" in item}
    
    enhanced_segments = []
    for i, seg in enumerate(segments):
        seg_copy = dict(seg)
        if i in corrections_map:
            corrected_text = corrections_map[i]
            if not corrected_text:
                continue  # Whole segment removed as filler
            seg_copy["text"] = corrected_text
            seg_copy["original_text"] = seg.get("text", "")
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

saved_key = load_saved_api_key()

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
    st.subheader("🤖 Google Gemini AI (Умная правка)")
    use_gemini = st.checkbox(
        "Исправлять ошибки и паразиты через Gemini",
        value=bool(saved_key),
        help="Использует нейросеть Gemini для контекстного исправления ошибок ASR, терминов, опечаток и удаления слов-паразитов"
    )
    
    gemini_api_key = ""
    gemini_model = "gemini-3.8-flash"
    if use_gemini:
        gemini_api_key = st.text_input(
            "Gemini API Key:",
            value=saved_key,
            type="password",
            help="Получите бесплатный ключ: https://aistudio.google.com/app/apikey"
        )
        if gemini_api_key and gemini_api_key != saved_key:
            save_api_key(gemini_api_key)
            st.success("API-ключ сохранён!")
            
        gemini_model = st.selectbox(
            "Модель Gemini:",
            options=["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
            index=0,
            help="gemini-3.8-flash — актуальная сверхбыстрая и качественная модель"
        )
    
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
        help="Удаляет точки, запятые, тире, кавычки и прочие знаки препинания из текста и субтитров"
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
    file_ext = uploaded_file.name.split('.')[-1].lower()
    base_name = os.path.splitext(uploaded_file.name)[0]
    total_start_time = time.time()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file_path = os.path.join(temp_dir, f"input.{file_ext}")
        wav_file_path = os.path.join(temp_dir, "audio.wav")
        
        with open(input_file_path, "wb") as f:
            f.write(uploaded_file.read())
            
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
                
                raw_segments = result.get("segments", [])
                
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
                
                # Optional Gemini AI enhancement
                gemini_applied = False
                if use_gemini and gemini_api_key.strip():
                    with st.spinner(f"✨ Обработка через Google Gemini ({gemini_model}): исправление ошибок и удаление паразитов..."):
                        try:
                            processed_segments = enhance_transcript_with_gemini(
                                processed_segments,
                                api_key=gemini_api_key.strip(),
                                model_name=gemini_model
                            )
                            gemini_applied = True
                        except Exception as ge:
                            st.warning(f"Внимание: ошибка Gemini API ({ge}). Использован локальный результат.")
                
                total_execution_time = time.time() - total_start_time
                
                # Performance metrics
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Извлечение аудио (FFmpeg)", f"{conversion_time:.2f} с")
                mcol2.metric("Время транскрибации AI", f"{transcription_time:.2f} с")
                mcol3.metric("Общее время работы", f"{total_execution_time:.2f} с")
                mcol4.metric("Gemini AI правка", "Включена ✅" if gemini_applied else "Выключена")
                
                if remove_fillers and removed_fillers_count > 0:
                    st.info(f"🚫 Локально удалено слов-паразитов: **{removed_fillers_count}**" + (" + дополнительная правка Gemini" if gemini_applied else ""))
                
                st.subheader("Результаты транскрибации")
                
                tab_names = ["💬 Предложения и сегменты", "⏱️ Пословные таймкоды", "🎬 Субтитры (SRT / VTT)"]
                if gemini_applied:
                    tab_names.append("🔄 Сравнение: Исходный vs Gemini")
                    
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
                
                # Tab 4: Diff comparison if Gemini applied
                if tab_diff is not None:
                    with tab_diff:
                        st.markdown("### Сравнение текста до и после обработки нейросетью Gemini")
                        diff_rows = []
                        for seg in processed_segments:
                            if "original_text" in seg:
                                diff_rows.append({
                                    "Таймкод": f"{format_time_display(seg['start'])} - {format_time_display(seg['end'])}",
                                    "Исходный текст (ASR)": seg["original_text"],
                                    "Исправлено Gemini AI": seg["text"]
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
* **Gemini AI Enhanced:** `{gemini_applied} ({gemini_model})`
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
                    "duration_seconds": result.get("duration_seconds"),
                    "gemini_enhanced": gemini_applied,
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
                st.error(f"Ошибка при транскрибации: {e}")
