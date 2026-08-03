import io
import asyncio
import base64
import hashlib
import shutil
import time
import uuid
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional

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
# key → edge-tts voice name.
# The original 12 keys (en-US-female, en-US-male, etc.) are UNCHANGED — any
# saved selectedVoice value in existing app installs keeps working identically.
# Everything below that is new, additive English variety.
VOICE_OPTIONS: dict[str, str] = {
    # ── Original 12 — unchanged ─────────────────────────────────────────────
    "en-US-female": "en-US-JennyNeural",
    "en-US-male":   "en-US-GuyNeural",
    "en-GB-female": "en-GB-SoniaNeural",
    "en-GB-male":   "en-GB-RyanNeural",
    "en-IN-female": "en-IN-NeerjaNeural",
    "en-IN-male":   "en-IN-PrabhatNeural",
    "en-AU-female": "en-AU-NatashaNeural",
    "en-AU-male":   "en-AU-WilliamNeural",
    "hi-female":    "hi-IN-SwaraNeural",
    "hi-male":      "hi-IN-MadhurNeural",
    "mr-female":    "mr-IN-AarohiNeural",
    "mr-male":      "mr-IN-ManoharNeural",

    # ── en-US — 14 additional confirmed voices ──────────────────────────────
    "en-US-aria-female":              "en-US-AriaNeural",
    "en-US-ana-female":               "en-US-AnaNeural",
    "en-US-andrew-male":              "en-US-AndrewNeural",
    "en-US-andrew-multilingual-male": "en-US-AndrewMultilingualNeural",
    "en-US-ava-female":               "en-US-AvaNeural",
    "en-US-ava-multilingual-female":  "en-US-AvaMultilingualNeural",
    "en-US-brian-male":               "en-US-BrianNeural",
    "en-US-brian-multilingual-male":  "en-US-BrianMultilingualNeural",
    "en-US-christopher-male":         "en-US-ChristopherNeural",
    "en-US-emma-multilingual-female": "en-US-EmmaMultilingualNeural",
    "en-US-eric-male":                "en-US-EricNeural",
    "en-US-michelle-female":          "en-US-MichelleNeural",
    "en-US-roger-male":               "en-US-RogerNeural",
    "en-US-steffan-male":             "en-US-SteffanNeural",

    # ── en-GB — 3 additional confirmed voices ───────────────────────────────
    "en-GB-libby-female":  "en-GB-LibbyNeural",
    "en-GB-maisie-female": "en-GB-MaisieNeural",
    "en-GB-thomas-male":   "en-GB-ThomasNeural",

    # ── Other English accents — one voice family each, all confirmed ───────
    "en-CA-clara-female":   "en-CA-ClaraNeural",
    "en-IE-emily-female":   "en-IE-EmilyNeural",
    "en-KE-asilia-female":  "en-KE-AsiliaNeural",
    "en-KE-chilemba-male":  "en-KE-ChilembaNeural",
    "en-NG-abeo-male":      "en-NG-AbeoNeural",
    "en-NG-ezinne-female":  "en-NG-EzinneNeural",
    "en-NZ-mitchell-male":  "en-NZ-MitchellNeural",
    "en-NZ-molly-female":   "en-NZ-MollyNeural",
    "en-PH-james-male":     "en-PH-JamesNeural",
    "en-PH-rosa-female":    "en-PH-RosaNeural",
    "en-SG-luna-female":    "en-SG-LunaNeural",
    "en-SG-wayne-male":     "en-SG-WayneNeural",
    "en-TZ-elimu-male":     "en-TZ-ElimuNeural",
    "en-TZ-imani-female":   "en-TZ-ImaniNeural",
    "en-ZA-leah-female":    "en-ZA-LeahNeural",
    "en-ZA-luke-male":      "en-ZA-LukeNeural",
    "en-HK-sam-male":       "en-HK-SamNeural",
    "en-HK-yan-female":     "en-HK-YanNeural",
}

DEFAULT_VOICE = "en-US-female"

# Fixed sample line for the voice-preview endpoint — same text every time
# means every preview after the first (per voice) is a pure cache hit.
PREVIEW_TEXT = "Hi there! This is a quick preview of how I sound when reading your text aloud."

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


def _resolve_voice_and_pitch(text: str, voice_key: str, pitch: str, gender: str) -> tuple[str, str, str]:
    """Shared validation used by text_to_wav, stream_audio_chunks, and
    text_to_wav_with_timings — keeps all three in sync on voice/pitch rules."""
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

    return voice_key, voice_name, _normalize_pitch(pitch)


def validate_stream_params(text: str, voice_key: str, gender: str) -> None:
    """
    Call this BEFORE creating a StreamingResponse around stream_audio_chunks.
    That's an async generator — calling it just creates the generator
    object, it doesn't run any code yet, so a ValueError inside it only
    fires once Starlette has already started sending a 200 response. That
    turns a validation error into a corrupted stream instead of a clean
    HTTP error. Calling this first, synchronously, in the route handler
    avoids that. Enforces the normal 50,000-char single-request cap.
    """
    _resolve_voice_and_pitch(text, voice_key, "+0Hz", gender)


def validate_chunked_params(text: str, voice_key: str, gender: str) -> None:
    """
    Same purpose as validate_stream_params, but for stream_pdf_audio —
    which is explicitly meant to handle text OVER 50,000 characters via
    chunking, so it must NOT apply that single-request cap. Only checks
    that the voice key is valid and text isn't empty.
    """
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")
    resolved_key = auto_select_voice(text, preferred_gender=gender) if voice_key == "auto" else voice_key
    if resolved_key not in VOICE_OPTIONS:
        raise ValueError(
            f"Unknown voice '{resolved_key}'. "
            f"Available: {list(VOICE_OPTIONS.keys())} or 'auto'"
        )


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
    voice_key, voice_name, pitch_hz = _resolve_voice_and_pitch(text, voice_key, pitch, gender)
    text = text.strip()

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
#  CHUNKED GENERATION — for long PDFs. Previously /tts/pdf handed the ENTIRE
#  extracted text straight to text_to_wav, which hard-rejects anything over
#  50,000 characters with a generic 422 — a ~40+ page PDF just failed outright
#  with no partial result. This splits on sentence boundaries, generates each
#  piece (still benefiting from the normal cache), and concatenates them into
#  one file — same approach the Android app already uses for its own
#  client-side chunking, so playback behavior stays consistent either way.
# ══════════════════════════════════════════════════════════════════════════════

def _split_text_for_chunking(text: str, max_chars: int = 4500) -> list[str]:
    """Splits on sentence boundaries where possible, falling back to a hard
    cut only for single sentences longer than max_chars themselves."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        piece = sentence if sentence.endswith((".", "!", "?")) else sentence + "."
        if len(piece) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(piece):
                end = min(start + max_chars, len(piece))
                if end < len(piece):
                    last_space = piece.rfind(" ", start, end)
                    if last_space > start:
                        end = last_space
                chunks.append(piece[start:end].strip())
                start = end
        elif len(current) + len(piece) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            current = piece
        else:
            current = f"{current} {piece}".strip()

    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


async def text_to_wav_chunked(
    text: str,
    voice_key: str = "auto",
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    gender: str = "female",
    max_chars: int = 4500,
) -> tuple[Path, str]:
    """
    Like text_to_wav, but handles text of ANY length by splitting into
    sentence-safe chunks and concatenating the results — no more hard 422
    on long PDFs. For text that already fits in one chunk, behaves
    identically to calling text_to_wav directly (single generation, normal
    caching, no concatenation overhead).
    """
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    if len(text) <= max_chars:
        return await text_to_wav(text, voice_key, rate, volume, pitch, gender)

    resolved_voice_key = auto_select_voice(text, preferred_gender=gender) if voice_key == "auto" else voice_key
    voice_name_check = VOICE_OPTIONS.get(resolved_voice_key)
    if not voice_name_check:
        raise ValueError(
            f"Unknown voice '{resolved_voice_key}'. "
            f"Available: {list(VOICE_OPTIONS.keys())} or 'auto'"
        )
    voice_key = resolved_voice_key
    pitch_hz = _normalize_pitch(pitch)

    # A repeat request for the exact same long text+voice+rate+pitch (e.g.
    # someone re-converts the same PDF) skips regeneration entirely.
    cached_combined = get_cached_file(text, voice_key, rate, pitch_hz)
    if cached_combined:
        return cached_combined, voice_key

    chunks = _split_text_for_chunking(text, max_chars)
    if not chunks:
        raise ValueError("Text produced no chunks after splitting.")

    chunk_paths: list[Path] = []
    used_voice = voice_key
    for chunk in chunks:
        chunk_path, used_voice = await text_to_wav(chunk, voice_key, rate, volume, pitch, gender)
        chunk_paths.append(chunk_path)

    combined = OUTPUT_DIR / f"{uuid.uuid4().hex}_combined.wav"
    with open(combined, "wb") as out:
        for chunk_path in chunk_paths:
            with open(chunk_path, "rb") as f:
                shutil.copyfileobj(f, out)
        # NOTE: chunk_paths are cache files (get_cached_file/save_to_cache
        # paths under CACHE_DIR) — do NOT delete them here, they're meant
        # to persist and be reused by future requests.

    # Cache the COMBINED result too (regardless of the 5,000-char cap that
    # applies to individual chunks — a full PDF's audio is exactly the kind
    # of expensive-to-regenerate result worth caching even if large).
    save_to_cache(text, voice_key, rate, pitch_hz, combined)

    logger.info(f"Chunked generation: {len(chunks)} chunks | voice={used_voice} | total_chars={len(text)}")
    return combined, used_voice


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMING — audio chunks yielded as edge-tts generates them, so a client
#  can start playing before the whole file exists. NOTE: edge-tts's underlying
#  wire format is MP3 (audio-24khz-48kbitrate-mono-mp3), not WAV — labeled
#  honestly as "audio/mpeg" here rather than perpetuating the existing
#  /tts/text endpoints' ".wav" filename (which works today only because
#  nothing downstream currently validates the container format).
# ══════════════════════════════════════════════════════════════════════════════

async def stream_audio_chunks(
    text: str,
    voice_key: str = "auto",
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    gender: str = "female",
) -> AsyncGenerator[bytes, None]:
    """
    Yields raw MP3 audio bytes as edge-tts produces them. Also writes the
    same bytes to the normal cache file as they stream by, so a second
    identical request (streamed OR non-streamed) still gets a cache hit —
    streaming doesn't opt this request out of caching, it just doesn't make
    the FIRST request wait for the full file before anything is sent.
    """
    voice_key, voice_name, pitch_hz = _resolve_voice_and_pitch(text, voice_key, pitch, gender)
    text = text.strip()

    cached = get_cached_file(text, voice_key, rate, pitch_hz)
    if cached:
        with open(cached, "rb") as f:
            while chunk := f.read(65536):
                yield chunk
        return

    communicate = edge_tts.Communicate(
        text=text, voice=voice_name, rate=rate, volume=volume, pitch=pitch_hz,
    )

    filename = OUTPUT_DIR / f"{uuid.uuid4().hex}.mp3"
    total_bytes = 0
    try:
        with open(filename, "wb") as out_file:
            async for message in communicate.stream():
                if message["type"] == "audio":
                    data = message["data"]
                    total_bytes += len(data)
                    out_file.write(data)
                    yield data
    except Exception as e:
        logger.error(f"edge-tts streaming error: {e}")
        filename.unlink(missing_ok=True)
        raise RuntimeError(f"TTS streaming failed: {e}")

    if total_bytes == 0:
        filename.unlink(missing_ok=True)
        raise RuntimeError("TTS produced no audio data.")

    if len(text) <= 5_000:
        save_to_cache(text, voice_key, rate, pitch_hz, filename)
    else:
        filename.unlink(missing_ok=True)  # don't leave large uncached files around

    logger.info(f"Streamed: voice={voice_key} | pitch={pitch_hz} | bytes={total_bytes} | chars={len(text)}")


# ══════════════════════════════════════════════════════════════════════════════
#  PDF STREAMING — combines chunking (arbitrary-length PDFs, same as
#  text_to_wav_chunked) with progressive streaming (playback can start on
#  the first chunk's audio while later chunks are still generating).
#
#  On a cache hit for the full text, this just streams the cached file
#  straight off disk — fast, and behaves like a normal file download from
#  the client's point of view. On a miss, chunks are generated and streamed
#  one after another; by the time the LAST chunk starts generating, the
#  FIRST chunk's audio has likely already finished playing client-side.
#  The complete result is written to a single file as it goes and cached
#  at the end — this is the "save it once it's done" behavior requested.
# ══════════════════════════════════════════════════════════════════════════════

async def stream_pdf_audio(
    text: str,
    voice_key: str = "auto",
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    gender: str = "female",
    max_chars: int = 4500,
) -> AsyncGenerator[bytes, None]:
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    resolved_voice_key = auto_select_voice(text, preferred_gender=gender) if voice_key == "auto" else voice_key
    voice_name = VOICE_OPTIONS.get(resolved_voice_key)
    if not voice_name:
        raise ValueError(
            f"Unknown voice '{resolved_voice_key}'. "
            f"Available: {list(VOICE_OPTIONS.keys())} or 'auto'"
        )
    voice_key = resolved_voice_key
    pitch_hz = _normalize_pitch(pitch)

    # Fast path: identical PDF+voice+rate+pitch requested before.
    cached = get_cached_file(text, voice_key, rate, pitch_hz)
    if cached:
        with open(cached, "rb") as f:
            while chunk := f.read(65536):
                yield chunk
        return

    chunks = _split_text_for_chunking(text, max_chars) if len(text) > max_chars else [text]
    if not chunks:
        raise ValueError("Text produced no chunks after splitting.")

    combined_path = OUTPUT_DIR / f"{uuid.uuid4().hex}_pdf_stream.mp3"
    total_bytes = 0

    try:
        with open(combined_path, "wb") as out_file:
            for chunk_text in chunks:
                communicate = edge_tts.Communicate(
                    text=chunk_text, voice=voice_name, rate=rate, volume=volume, pitch=pitch_hz,
                )
                async for message in communicate.stream():
                    if message["type"] == "audio":
                        data = message["data"]
                        total_bytes += len(data)
                        out_file.write(data)
                        yield data  # sent to the client immediately, chunk by chunk
    except Exception as e:
        logger.error(f"PDF streaming error: {e}")
        combined_path.unlink(missing_ok=True)
        raise RuntimeError(f"PDF audio streaming failed: {e}")

    if total_bytes == 0:
        combined_path.unlink(missing_ok=True)
        raise RuntimeError("TTS produced no audio data.")

    # Cache the complete result regardless of length — this is the
    # "save once complete" behavior. A repeat request for this exact PDF
    # (same extracted text+voice+rate+pitch) hits the fast path above.
    save_to_cache(text, voice_key, rate, pitch_hz, combined_path)

    logger.info(
        f"PDF streamed: {len(chunks)} chunk(s) | voice={voice_key} | "
        f"pitch={pitch_hz} | bytes={total_bytes} | chars={len(text)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  WORD-TIMED AUDIO — for a future scrubbable "listen while reading" player.
#  Returns the full audio PLUS a list of {text, offset_ms, duration_ms} for
#  every word, from edge-tts's WordBoundary events. Not wired into the app
#  yet — building it now while the rest of the API is fresh.
# ══════════════════════════════════════════════════════════════════════════════

async def text_to_wav_with_timings(
    text: str,
    voice_key: str = "auto",
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    gender: str = "female",
) -> tuple[Path, str, list[dict]]:
    """
    Same as text_to_wav, but also captures word-level timing. Not served
    from the audio cache (timings aren't cached today) — always regenerates.
    Returns (Path to audio file, resolved voice_key, word_timings list).
    Each timing entry: {"text": str, "offset_ms": float, "duration_ms": float}.
    """
    voice_key, voice_name, pitch_hz = _resolve_voice_and_pitch(text, voice_key, pitch, gender)
    text = text.strip()

    communicate = edge_tts.Communicate(
        text=text, voice=voice_name, rate=rate, volume=volume, pitch=pitch_hz,
    )

    filename = OUTPUT_DIR / f"{uuid.uuid4().hex}.mp3"
    word_timings: list[dict] = []
    try:
        with open(filename, "wb") as out_file:
            async for message in communicate.stream():
                if message["type"] == "audio":
                    out_file.write(message["data"])
                elif message["type"] == "WordBoundary":
                    # edge-tts reports offset/duration in 100-nanosecond ticks.
                    word_timings.append({
                        "text": message["text"],
                        "offset_ms": message["offset"] / 10_000,
                        "duration_ms": message["duration"] / 10_000,
                    })
    except Exception as e:
        logger.error(f"edge-tts timed-generation error: {e}")
        filename.unlink(missing_ok=True)
        raise RuntimeError(f"TTS generation failed: {e}")

    if not filename.exists() or filename.stat().st_size == 0:
        raise RuntimeError("TTS produced an empty file.")

    logger.info(
        f"Generated with timings: voice={voice_key} | pitch={pitch_hz} | "
        f"words={len(word_timings)} | chars={len(text)}"
    )
    return filename, voice_key, word_timings



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