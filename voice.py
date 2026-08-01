import io
import asyncio
import hashlib
import shutil
import time
import uuid
import logging
from pathlib import Path
from typing import Optional

import edge_tts
import PyPDF2

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("generated_audio")
OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path("audio_cache")
CACHE_DIR.mkdir(exist_ok=True)

FILE_TTL_SECONDS  = 600    # 10 min — generated files
CACHE_TTL_SECONDS = 3600   # 1 hour — cached files

# ── Voice table ───────────────────────────────────────────────────────────────
# key → edge-tts voice name
VOICE_OPTIONS: dict[str, str] = {
    # English
    "en-US-female": "en-US-JennyNeural",
    "en-US-male":   "en-US-GuyNeural",
    "en-GB-female": "en-GB-SoniaNeural",
    "en-GB-male":   "en-GB-RyanNeural",
    "en-IN-female": "en-IN-NeerjaNeural",
    "en-IN-male":   "en-IN-PrabhatNeural",
    "en-AU-female": "en-AU-NatashaNeural",
    "en-AU-male":   "en-AU-WilliamNeural",
    # Hindi
    "hi-female":    "hi-IN-SwaraNeural",
    "hi-male":      "hi-IN-MadhurNeural",
    # Marathi
    "mr-female":    "mr-IN-AarohiNeural",
    "mr-male":      "mr-IN-ManoharNeural",
}

DEFAULT_VOICE = "en-US-female"

LANG_DEFAULT_VOICE: dict[str, str] = {
    "hindi":   "hi-female",
    "marathi": "mr-female",
    "english": "en-IN-female",
}


# ══════════════════════════════════════════════════════════════════════════════
#  LANGUAGE DETECTION  (unchanged from your original — pure Python, no ML)
# ══════════════════════════════════════════════════════════════════════════════

def _count_script_chars(text: str) -> dict[str, int]:
    counts = {"devanagari": 0, "latin": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
            counts["latin"] += 1
        elif ch.isalpha():
            counts["other"] += 1
    return counts


_MARATHI_MARKERS = {"\u0933", "\u0965"}
_MARATHI_WORDS = {
    "आहे", "नाही", "आणि", "हे", "ते", "मला", "तुम्ही", "आम्ही",
    "काय", "कसे", "होते", "असे", "येथे", "त्यांनी", "केले", "झाले",
    "मराठी", "महाराष्ट्र",
}
_HINDI_WORDS = {
    "है", "नहीं", "और", "यह", "वह", "मुझे", "आप", "हम",
    "क्या", "कैसे", "था", "ऐसे", "यहाँ", "उन्होंने", "किया", "हुआ",
    "हिंदी", "भारत",
}


def detect_language(text: str) -> str:
    sample = text[:2000]
    counts = _count_script_chars(sample)
    total_alpha = counts["devanagari"] + counts["latin"] + counts["other"]

    if total_alpha == 0:
        return "english"

    devanagari_ratio = counts["devanagari"] / total_alpha
    if devanagari_ratio < 0.20:
        return "english"

    words_in_text = set(sample.split())
    chars_in_text = set(sample)

    marathi_score = (
        len(words_in_text & _MARATHI_WORDS) * 2 +
        len(chars_in_text & _MARATHI_MARKERS) * 3
    )
    hindi_score = len(words_in_text & _HINDI_WORDS) * 2

    if marathi_score > hindi_score:
        return "marathi"
    elif hindi_score > marathi_score:
        return "hindi"
    else:
        if "\u0933" in sample:
            return "marathi"
        return "hindi"


def auto_select_voice(text: str, preferred_gender: str = "female") -> str:
    lang = detect_language(text)
    gender_suffix = "female" if preferred_gender == "female" else "male"
    voice_map = {
        "hindi":   f"hi-{gender_suffix}",
        "marathi": f"mr-{gender_suffix}",
        "english": f"en-IN-{gender_suffix}",
    }
    voice_key = voice_map.get(lang, DEFAULT_VOICE)
    logger.info(f"Auto-detected language: '{lang}' → voice: '{voice_key}'")
    return voice_key


# ══════════════════════════════════════════════════════════════════════════════
#  PDF EXTRACTION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t.strip())
        full_text = "\n".join(pages_text)
        if not full_text.strip():
            raise ValueError("PDF appears to be empty or contains only images.")
        return full_text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
#  CACHE
# ══════════════════════════════════════════════════════════════════════════════
# NOTE: cache key now includes pitch and rate — previously it only hashed
# text+voice, so two requests for the same text/voice but DIFFERENT pitch or
# rate would have silently served each other's cached audio. That bug is now
# fixed as part of adding real pitch support.

def _cache_key(text: str, voice_key: str, rate: str, pitch: str) -> str:
    raw = f"{voice_key}::{rate}::{pitch}::{text}"
    return hashlib.sha256(raw.encode()).hexdigest()

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.wav"

def get_cached_file(text: str, voice_key: str, rate: str, pitch: str) -> Optional[Path]:
    key = _cache_key(text, voice_key, rate, pitch)
    path = _cache_path(key)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            logger.info(f"Cache HIT: {key[:12]}...")
            return path
        path.unlink(missing_ok=True)
    return None

def save_to_cache(text: str, voice_key: str, rate: str, pitch: str, source_path: Path) -> None:
    key = _cache_key(text, voice_key, rate, pitch)
    shutil.copy2(source_path, _cache_path(key))
    logger.info(f"Cached: {key[:12]}...")


# ══════════════════════════════════════════════════════════════════════════════
#  CORE TTS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_pitch(pitch: str) -> str:
    """
    Accepts either edge-tts's native '+5Hz'/'-10Hz' format, or a bare
    number/percentage from a UI slider (e.g. "1.2" meaning 1.2x, matching
    how the Android app's pitch slider is scaled 0.5-2.0) and converts it
    to a safe Hz string edge-tts understands. Defaults to no shift.
    """
    pitch = (pitch or "").strip()
    if not pitch:
        return "+0Hz"
    if pitch.endswith("Hz"):
        return pitch  # already in edge-tts's native format

    try:
        # Treat a bare float as the app's 0.5-2.0 slider scale, map to
        # a modest ±20Hz range so it stays natural-sounding rather than
        # cartoonish at the extremes.
        value = float(pitch)
        hz = round((value - 1.0) * 20)
        hz = max(-20, min(20, hz))
        sign = "+" if hz >= 0 else ""
        return f"{sign}{hz}Hz"
    except ValueError:
        return "+0Hz"


async def text_to_wav(
    text: str,
    voice_key: str = "auto",
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    gender: str = "female",
) -> tuple[Path, str]:
    """
    Convert text → WAV using edge-tts.
    Returns (Path to WAV file, resolved voice_key that was used).
    pitch accepts either native "+5Hz" format or a 0.5-2.0 slider float.
    """
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")
    if len(text) > 50_000:
        raise ValueError("Text exceeds 50,000 character limit.")

    if voice_key == "auto":
        voice_key = auto_select_voice(text, preferred_gender=gender)

    voice_name = VOICE_OPTIONS.get(voice_key)
    if not voice_name:
        raise ValueError(
            f"Unknown voice '{voice_key}'. "
            f"Available: {list(VOICE_OPTIONS.keys())} or 'auto'"
        )

    pitch_hz = _normalize_pitch(pitch)

    cached = get_cached_file(text, voice_key, rate, pitch_hz)
    if cached:
        return cached, voice_key

    filename = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"
    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_name,
            rate=rate,
            volume=volume,
            pitch=pitch_hz,
        )
        await communicate.save(str(filename))
    except Exception as e:
        logger.error(f"edge-tts error: {e}")
        filename.unlink(missing_ok=True)
        raise RuntimeError(f"TTS generation failed: {e}")

    if not filename.exists() or filename.stat().st_size == 0:
        raise RuntimeError("TTS produced an empty file.")

    if len(text) <= 5_000:
        save_to_cache(text, voice_key, rate, pitch_hz, filename)

    logger.info(f"Generated: {filename.name} | voice={voice_key} | pitch={pitch_hz} | chars={len(text)}")
    return filename, voice_key


# ══════════════════════════════════════════════════════════════════════════════
#  GARBAGE COLLECTION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def run_garbage_collection() -> int:
    now = time.time()
    deleted = 0
    for f in OUTPUT_DIR.glob("*.wav"):
        try:
            if now - f.stat().st_mtime > FILE_TTL_SECONDS:
                f.unlink()
                deleted += 1
        except Exception as e:
            logger.warning(f"GC error on {f.name}: {e}")
    if deleted:
        logger.info(f"GC: removed {deleted} expired file(s)")
    return deleted

async def scheduled_gc(interval: int = 300):
    while True:
        await asyncio.sleep(interval)
        run_garbage_collection()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS FOR ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def list_voices() -> dict:
    result = {}
    for key, val in VOICE_OPTIONS.items():
        parts = key.split("-")
        lang_code = parts[0]
        gender = parts[-1]
        lang_label = {"en": "English", "hi": "Hindi", "mr": "Marathi"}.get(lang_code, lang_code)
        result[key] = {
            "voice_id": val,
            "language": lang_label,
            "language_code": lang_code,
            "gender": gender,
        }
    return result
