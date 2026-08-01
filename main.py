import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from voice import (
    text_to_wav,
    extract_text_from_pdf,
    list_voices,
    run_garbage_collection,
    scheduled_gc,
    detect_language,
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

@app.post("/tts/text", tags=["TTS"], summary="Text → WAV (JSON body)")
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


@app.post("/tts/text/form", tags=["TTS"], summary="Text → WAV (form-data)")
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


@app.post("/tts/pdf", tags=["TTS"], summary="PDF → WAV")
async def tts_from_pdf(
    background_tasks: BackgroundTasks,
    file:   UploadFile = File(...),
    voice:  str = Form("auto"),
    gender: str = Form("female"),
    rate:   str = Form("+0%"),
    volume: str = Form("+0%"),
    pitch:  str = Form("+0Hz"),
):
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
        wav_path, used_voice = await text_to_wav(
            text=text, voice_key=voice, rate=rate, volume=volume, pitch=pitch, gender=gender,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _wav_response(wav_path, used_voice, background_tasks)
