import asyncio
import base64
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from voice import (
    text_to_wav,
    text_to_wav_chunked,
    stream_audio_chunks,
    stream_pdf_audio,
    text_to_wav_with_timings,
    extract_text_from_pdf,
    list_voices,
    run_garbage_collection,
    scheduled_gc,
    detect_language,
    validate_stream_params,
    validate_chunked_params,
    PREVIEW_TEXT,
)

# ══════════════════════════════════════════════════════════════════
#  LIFESPAN
# ══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    gc_task = asyncio.create_task(scheduled_gc(interval=300))
    yield
    gc_task.cancel()
    try:
        await gc_task
    except asyncio.CancelledError:
        pass


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="VoxLibro TTS API",
    description=(
        "Convert plain text or PDF to natural-sounding WAV audio via edge-tts. "
        "Supports English, Hindi, and Marathi with automatic language detection, "
        "plus rate/volume/pitch control. This is the single consolidated backend — "
        "the old two-API (primary + fallback) split has been retired."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://voxlibro.onrender.com",
        "https://voxlibro.netlify.app",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_origin_regex=r"http://localhost(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
#  ABUSE PROTECTION — rate limiting + shared-secret header
# ══════════════════════════════════════════════════════════════════
# Set VOXLIBRO_API_SECRET as an environment variable in the Render
# dashboard (Environment tab) — NOT hardcoded here, and NOT committed to
# the repo. The Android app must send the same value in the
# "X-VoxLibro-Key" header on every /tts/* request.
#
# If the env var is left unset, the secret check is skipped entirely —
# this is intentional so local development / testing via /docs still
# works without needing the header. Once you set the env var in Render,
# enforcement turns on automatically with no code change.
_API_SECRET = os.environ.get("VOXLIBRO_API_SECRET", "")

RATE_LIMIT_MAX_REQUESTS = 20   # per IP
RATE_LIMIT_WINDOW_SECONDS = 60

# In-memory sliding window — fine for a single Render instance. If this
# ever runs on multiple instances behind a load balancer, this would need
# to move to something shared (e.g. Redis) since each instance would
# otherwise track its own separate counts.
_request_log: dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    # Render sits behind a proxy — the real client IP arrives via
    # X-Forwarded-For, not request.client.host (which would be the proxy).
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def verify_request(request: Request) -> None:
    """Dependency applied to every TTS-consuming route: checks the shared
    secret (if configured) and enforces per-IP rate limiting."""
    if _API_SECRET:
        provided = request.headers.get("x-voxlibro-key", "")
        if provided != _API_SECRET:
            raise HTTPException(status_code=401, detail="Missing or invalid API key.")

    ip = _client_ip(request)
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    recent = [t for t in _request_log[ip] if t > window_start]
    if len(recent) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded — max {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
        )

    recent.append(now)
    _request_log[ip] = recent


# ══════════════════════════════════════════════════════════════════
#  REQUEST MODELS
# ══════════════════════════════════════════════════════════════════
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    voice: str = Field(
        "auto",
        description=(
            "Voice key from /voices, or 'auto' to detect language automatically. "
            "Hindi text → hi-IN voice, Marathi → mr-IN voice, English → en-IN voice."
        ),
    )
    gender: str = Field("female", description="'female' or 'male' — used when voice='auto'")
    rate: str   = Field("+0%",    description="Speed:  +10%, -20%, etc.")
    volume: str = Field("+0%",    description="Volume: +5%,  -10%, etc.")
    pitch: str  = Field(
        "+0Hz",
        description=(
            "Pitch shift. Accepts edge-tts native format ('+5Hz', '-10Hz') "
            "OR a bare 0.5-2.0 float matching the app's slider scale (1.0 = no change)."
        ),
    )


# ══════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════
def _cleanup(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

def _wav_response(wav_path: Path, used_voice: str, bg: BackgroundTasks) -> FileResponse:
    bg.add_task(_cleanup, wav_path)
    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename="output.wav",
        headers={"X-Voice-Used": used_voice},
    )


# ══════════════════════════════════════════════════════════════════
#  UTILITY ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Utility"], summary="API info")
async def root():
    return {
        "name":    "VoxLibro TTS API",
        "version": "3.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "voices":  "/voices",
    }


@app.get("/health", tags=["Utility"], summary="Health check")
async def health_check():
    """Returns 200 OK if the API is running. Use this as Render's health-check URL,
    and as the target for any external uptime-monitor keep-alive ping."""
    return {"status": "ok", "message": "VoxLibro TTS API is running."}


@app.get("/voices", tags=["Utility"], summary="List all available voices")
async def get_voices():
    voices = list_voices()
    grouped: dict = {}
    for key, info in voices.items():
        grouped.setdefault(info["language"], {})[key] = info
    return {"voices": voices, "grouped": grouped}


@app.post("/detect-language", tags=["Utility"], summary="Detect language of text")
async def detect_lang(body: dict):
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="'text' field is required.")
    lang = detect_language(text)
    return {
        "detected_language": lang,
        "recommended_voices": {
            "female": {"hindi": "hi-female", "marathi": "mr-female", "english": "en-IN-female"}[lang],
            "male":   {"hindi": "hi-male",   "marathi": "mr-male",   "english": "en-IN-male"}[lang],
        },
    }


@app.post("/admin/gc", tags=["Utility"], summary="Trigger garbage collection")
async def trigger_gc():
    deleted = run_garbage_collection()
    return {"deleted_files": deleted, "message": f"Removed {deleted} expired file(s)."}


# ══════════════════════════════════════════════════════════════════
#  TTS ROUTES
# ══════════════════════════════════════════════════════════════════

@app.post("/tts/text", tags=["TTS"], summary="Text → WAV (JSON body)", dependencies=[Depends(verify_request)])
async def tts_from_text(request: TTSRequest, background_tasks: BackgroundTasks):
    try:
        wav_path, used_voice = await text_to_wav(
            text=request.text,
            voice_key=request.voice,
            rate=request.rate,
            volume=request.volume,
            pitch=request.pitch,
            gender=request.gender,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _wav_response(wav_path, used_voice, background_tasks)


@app.post("/tts/text/form", tags=["TTS"], summary="Text → WAV (form-data)", dependencies=[Depends(verify_request)])
async def tts_from_text_form(
    background_tasks: BackgroundTasks,
    text:   str = Form(...),
    voice:  str = Form("auto"),
    gender: str = Form("female"),
    rate:   str = Form("+0%"),
    volume: str = Form("+0%"),
    pitch:  str = Form("+0Hz"),
):
    """
    This is the endpoint the Android app calls. `pitch` is new — the app was
    previously not sending it at all, so the UI's pitch slider had no effect
    in online mode. Accepts either "+5Hz" or a bare 0.5-2.0 float.
    """
    try:
        wav_path, used_voice = await text_to_wav(
            text=text, voice_key=voice, rate=rate, volume=volume, pitch=pitch, gender=gender,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _wav_response(wav_path, used_voice, background_tasks)


@app.post("/tts/stream", tags=["TTS"], summary="Text → streaming MP3 audio", dependencies=[Depends(verify_request)])
async def tts_stream(
    text:   str = Form(...),
    voice:  str = Form("auto"),
    gender: str = Form("female"),
    rate:   str = Form("+0%"),
    volume: str = Form("+0%"),
    pitch:  str = Form("+0Hz"),
):
    """
    True progressive streaming — audio chunks are sent to the client as
    edge-tts generates them, rather than waiting for the whole file. This is
    for perceived-speed on long texts: a client that plays chunks as they
    arrive can start audio well before generation finishes.

    Content-Type is honestly "audio/mpeg" — edge-tts's underlying wire
    format is MP3, not WAV, regardless of what the other /tts/text* routes'
    filenames suggest.

    NOTE: this is API-level capability only. Seeing the speed benefit
    requires the client to consume and play the stream progressively
    (e.g. ExoPlayer/MediaPlayer pointed at this URL) — a plain "wait for the
    whole response then play" client gets no benefit over /tts/text/form.
    """
    try:
        validate_stream_params(text, voice, gender)
        generator = stream_audio_chunks(
            text=text, voice_key=voice, rate=rate, volume=volume, pitch=pitch, gender=gender,
        )
        return StreamingResponse(generator, media_type="audio/mpeg")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/text/timed", tags=["TTS"], summary="Text → audio + word-level timings (JSON)", dependencies=[Depends(verify_request)])
async def tts_text_timed(
    background_tasks: BackgroundTasks,
    text:   str = Form(...),
    voice:  str = Form("auto"),
    gender: str = Form("female"),
    rate:   str = Form("+0%"),
    volume: str = Form("+0%"),
    pitch:  str = Form("+0Hz"),
):
    """
    Returns a single JSON payload with base64-encoded audio AND a
    word-by-word timing map (offset_ms/duration_ms per word), sourced from
    edge-tts's WordBoundary events.

    Built for a future "listen while reading" scrubbable player — NOT
    currently called by the app. Not streaming (defeats the purpose here:
    the client needs the complete timing map upfront to build a scrubber),
    and not cached (timings aren't part of today's cache key scheme).

    Response shape:
    {
      "voice_used": "en-US-female",
      "audio_base64": "...",
      "audio_format": "audio/mpeg",
      "word_timings": [{"text": "Hello", "offset_ms": 12.5, "duration_ms": 340.0}, ...]
    }
    """
    try:
        audio_path, used_voice, word_timings = await text_to_wav_with_timings(
            text=text, voice_key=voice, rate=rate, volume=volume, pitch=pitch, gender=gender,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    audio_bytes = audio_path.read_bytes()
    background_tasks.add_task(_cleanup, audio_path)

    return {
        "voice_used": used_voice,
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "audio_format": "audio/mpeg",
        "word_timings": word_timings,
    }


@app.post("/tts/pdf", tags=["TTS"], summary="PDF → WAV", dependencies=[Depends(verify_request)])
async def tts_from_pdf(
    background_tasks: BackgroundTasks,
    file:   UploadFile = File(...),
    voice:  str = Form("auto"),
    gender: str = Form("female"),
    rate:   str = Form("+0%"),
    volume: str = Form("+0%"),
    pitch:  str = Form("+0Hz"),
):
    """
    Upload a PDF file → API extracts text → returns audio. Any length is
    now accepted — long PDFs are split into sentence-safe chunks and
    concatenated server-side instead of hard-rejecting anything over
    50,000 characters like before. Scanned/image-only PDFs are still
    unsupported (no OCR).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        text = extract_text_from_pdf(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {e}")

    try:
        wav_path, used_voice = await text_to_wav_chunked(
            text=text, voice_key=voice, rate=rate, volume=volume, pitch=pitch, gender=gender,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _wav_response(wav_path, used_voice, background_tasks)


@app.post("/tts/pdf/stream", tags=["TTS"], summary="PDF → streaming MP3 audio", dependencies=[Depends(verify_request)])
async def tts_pdf_stream(
    file:   UploadFile = File(...),
    voice:  str = Form("auto"),
    gender: str = Form("female"),
    rate:   str = Form("+0%"),
    volume: str = Form("+0%"),
    pitch:  str = Form("+0Hz"),
):
    """
    Upload a PDF → audio streams back progressively as it generates,
    instead of waiting for the whole document to finish before anything is
    sent. Handles any PDF length (chunked internally, same as /tts/pdf) —
    by the time later pages are still generating, earlier pages' audio has
    likely already reached the client.

    The complete result is cached automatically once generation finishes
    (same cache the non-streaming endpoints use) — a repeat request for
    the identical PDF+voice+rate+pitch streams back instantly from cache
    rather than regenerating.

    NOTE: seeing the "starts speaking immediately" benefit requires the
    client to play the stream progressively (e.g. ExoPlayer/MediaPlayer
    pointed at this URL) rather than buffering the full response first —
    this is the API side of that; the Android app doesn't consume it this
    way yet.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        text = extract_text_from_pdf(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {e}")

    try:
        validate_chunked_params(text, voice, gender)
        generator = stream_pdf_audio(
            text=text, voice_key=voice, rate=rate, volume=volume, pitch=pitch, gender=gender,
        )
        return StreamingResponse(generator, media_type="audio/mpeg")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/preview/{voice_key}", tags=["TTS"], summary="Quick voice preview", dependencies=[Depends(verify_request)])
async def tts_preview(voice_key: str, background_tasks: BackgroundTasks):
    """
    Plays a short, fixed sample line in the requested voice — for a voice
    picker UI where users tap through options and want to hear each one
    instantly rather than waiting on a full generation round-trip.
    Always uses PREVIEW_TEXT at default rate/pitch, so after the very
    first request for any given voice, every subsequent preview of that
    same voice is a pure cache hit (no edge-tts network call at all).
    """
    try:
        wav_path, used_voice = await text_to_wav(
            text=PREVIEW_TEXT, voice_key=voice_key, rate="+0%", volume="+0%", pitch="+0Hz", gender="female",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _wav_response(wav_path, used_voice, background_tasks)