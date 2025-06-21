"""
Main FastAPI application for VoiceAI TTS Server with authentication and user management.
"""
import os
import io
import logging
import logging.handlers
import shutil
import time
import uuid
import yaml
import numpy as np
import librosa
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Literal
import webbrowser
import threading
import uvicorn
from fastapi import Request


from fastapi import (
    FastAPI, HTTPException, Request, File, UploadFile, Form, BackgroundTasks,
    Depends, Security
)
from fastapi.responses import (
    HTMLResponse, JSONResponse, StreamingResponse, FileResponse, RedirectResponse
)
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Internal imports
from config import (
    config_manager, get_host, get_port, get_log_file_path,
    get_output_path, get_reference_audio_path, get_predefined_voices_path,
    get_ui_title, get_gen_default_temperature, get_gen_default_exaggeration,
    get_gen_default_cfg_weight, get_gen_default_seed, get_gen_default_speed_factor,
    get_gen_default_language, get_audio_sample_rate, get_full_config_for_template,
    get_audio_output_format
)
import engine
from models import CustomTTSRequest, ErrorResponse, UpdateStatusResponse
import utils
from database import get_db, User, UsageLog, UserSession, GenerationQueue, init_database
from auth import auth_handler, create_user_session
from api import router as api_router

# OpenAI-compatible request model
class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"
    speed: float = 1.0
    seed: Optional[int] = None

# Logging setup
log_file_path_obj = get_log_file_path()
log_file_max_size_mb = config_manager.get_int("server.log_file_max_size_mb", 10)
log_backup_count = config_manager.get_int("server.log_file_backup_count", 5)

log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            str(log_file_path_obj),
            maxBytes=log_file_max_size_mb * 1024 * 1024,
            backupCount=log_backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Global variables
startup_complete_event = threading.Event()

def _delayed_browser_open(host: str, port: int):
    try:
        startup_complete_event.wait(timeout=30)
        if not startup_complete_event.is_set():
            logger.warning("Server startup timeout. Browser will not open automatically.")
            return
        time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Opening browser to: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser: {e}", exc_info=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VoiceAI Server: Initializing...")
    try:
        # Initialize database
        init_database()
        logger.info("Database initialized successfully")
        
        # Initialize directories
        paths_to_ensure = [
            config_manager.get_path("paths.output", "./outputs", ensure_absolute=True),
            config_manager.get_path("paths.reference_audio", "./reference_audio", ensure_absolute=True),
            config_manager.get_path("paths.predefined_voices", "./voices", ensure_absolute=True),
            config_manager.get_path("paths.model_cache", "./model_cache", ensure_absolute=True),
            Path("ui")
        ]
        for p in paths_to_ensure:
            p.mkdir(parents=True, exist_ok=True)

        if not engine.load_model():
            logger.critical("CRITICAL: TTS Model failed to load.")
        else:
            logger.info("TTS Model loaded successfully")
            host_address = get_host()
            server_port = get_port()
            browser_thread = threading.Thread(
                target=lambda: _delayed_browser_open(host_address, server_port),
                daemon=True
            )
            browser_thread.start()

        logger.info("Startup complete")
        startup_complete_event.set()
        yield
    except Exception as e:
        logger.error(f"FATAL ERROR during startup: {e}", exc_info=True)
        startup_complete_event.set()
        yield
    finally:
        logger.info("Server shutdown initiated")

# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)

# FastAPI Application
app = FastAPI(
    title="VoiceAI TTS Server",
    description="Text-to-Speech server with user management and authentication",
    version="2.0.2",
    lifespan=lifespan,
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def inject_auth_header_from_cookie(request: Request, call_next):
    # if no Authorization header but we have an access_token cookie…
    if "authorization" not in request.headers and "access_token" in request.cookies:
        token = request.cookies["access_token"]
        # Create a mutable copy of headers
        headers = list(request.scope["headers"])
        headers.append((b"authorization", f"Bearer {token}".encode()))
        request.scope["headers"] = headers
    return await call_next(request)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Exception handler for authentication redirects
@app.exception_handler(FastAPIHTTPException)
async def auth_exception_handler(request: Request, exc: FastAPIHTTPException):
    """Handle authentication exceptions by redirecting to login page for HTML requests"""
    if exc.status_code == 401:
        # Check if this is a browser request (HTML) vs API request (JSON)
        accept_header = request.headers.get("accept", "")
        user_agent = request.headers.get("user-agent", "")
        
        # More comprehensive check for browser requests
        is_browser_request = (
            "text/html" in accept_header or 
            "Mozilla" in user_agent or
            request.url.path in ["/", "/admin"] or
            not request.url.path.startswith("/api/")
        )
        
        if is_browser_request:
            # Redirect to login page for browser requests
            return RedirectResponse(url="/login", status_code=303)
        else:
            # Return JSON error for API requests
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )
    
    # For other HTTP exceptions, return the default response
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Mount static files and directories
ui_static_path = Path(__file__).parent / "ui"
if ui_static_path.is_dir():
    app.mount("/ui", StaticFiles(directory=ui_static_path), name="ui_static_assets")
    if (ui_static_path / "vendor").is_dir():
        app.mount("/vendor", StaticFiles(directory=ui_static_path / "vendor"), name="vendor_files")
    else:
        logger.warning(f"Vendor directory not found at '{ui_static_path}/vendor'. Wavesurfer might not load.")
else:
    logger.warning(f"UI static assets directory not found at '{ui_static_path}'. UI may not load correctly.")

# Mount outputs directory for generated audio files
outputs_static_path = get_output_path(ensure_absolute=True)
try:
    app.mount("/outputs", StaticFiles(directory=str(outputs_static_path)), name="generated_outputs")
except RuntimeError as e_mount_outputs:
    logger.error(
        f"Failed to mount /outputs directory '{outputs_static_path}': {e_mount_outputs}. "
        "Output files may not be accessible via URL."
    )

templates = Jinja2Templates(directory=str(ui_static_path))

# Include API router
app.include_router(api_router, prefix="/api")

# Static file routes
@app.get("/styles.css", include_in_schema=False)
async def get_styles():
    styles_file = ui_static_path / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404, detail="styles.css not found")

@app.get("/script.js", include_in_schema=False)
async def get_script():
    script_file = ui_static_path / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404, detail="script.js not found")

# Main routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request):
    """Serves the main web interface (index.html)."""
    logger.info("Request received for main UI page ('/').")
    
    # Check authentication manually to handle redirects properly
    try:
        # Try to get token from cookie first, then from Authorization header
        token = request.cookies.get("access_token")
        if not token:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
        
        if not token:
            logger.info("No authentication token found, redirecting to login")
            return RedirectResponse(url="/login", status_code=303)
        
        try:
            # Validate token and get user
            payload = auth_handler.decode_token(token)
            user_id = payload.get("sub")
            if not user_id:
                logger.debug("Invalid token format, redirecting to login")
                return RedirectResponse(url="/login", status_code=303)
            
            # Get database session and user
            db = next(get_db())
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                logger.debug("User not found, redirecting to login")
                return RedirectResponse(url="/login", status_code=303)
            
            if not user.is_active:
                logger.debug("User account disabled")
                return templates.TemplateResponse(
                    "login.html",
                    {
                        "request": request,
                        "error_message": "Your account has been disabled. Please contact an administrator."
                    }
                )
            
            if user.is_expired():
                logger.debug("User account expired")
                return templates.TemplateResponse(
                    "login.html",
                    {
                        "request": request,
                        "error_message": "Your account has expired. Please contact an administrator to renew."
                    }
                )
            
            # User is authenticated, serve the main UI
            return templates.TemplateResponse("index.html", {
                "request": request,
                "user": user
            })
            
        except Exception as auth_error:
            logger.debug(f"Authentication validation failed: {auth_error}")
            return RedirectResponse(url="/login", status_code=303)
            
    except Exception as e_render:
        logger.error(f"Error rendering main UI page: {e_render}", exc_info=True)
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1><p>Could not load the TTS interface. "
            "Please check server logs for more details.</p></body></html>",
            status_code=500,
        )

@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def get_login_page(request: Request):
    """Serves the login page."""
    # If user is already logged in, redirect to main page
    token = request.cookies.get("access_token")
    if token:
        try:
            payload = auth_handler.decode_token(token)
            user_id = payload.get("sub")
            if user_id:
                # Verify user still exists and is active
                db = next(get_db())
                user = db.query(User).filter(User.id == user_id).first()
                if user and user.is_active and not user.is_expired():
                    logger.info("User already authenticated, redirecting to main page")
                    return RedirectResponse(url="/", status_code=303)
        except Exception as e:
            logger.debug(f"Token validation failed during login page access: {e}")
            # Don't clear cookies here, just serve login page
            pass
    
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def get_admin_page(current_user: User = Depends(auth_handler.get_current_admin_user)):
    """Serves the admin page (admin users only)."""
    return templates.TemplateResponse("admin.html", {"request": {}})

# TTS Generation endpoint with user validation, concurrency control, usage tracking, and performance monitoring
@app.post(
    "/tts",
    tags=["TTS Generation"],
    summary="Generate speech with custom parameters",
    responses={
        200: {
            "content": {"audio/wav": {}, "audio/opus": {}, "audio/mp3": {}},
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        404: {
            "model": ErrorResponse,
            "description": "Required resource not found (e.g., voice file).",
        },
        403: {
            "model": ErrorResponse,
            "description": "Monthly character limit exceeded or concurrent request limit reached.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during generation.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
async def tts_endpoint(
    request: CustomTTSRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(auth_handler.get_current_user),
    db: Session = Depends(get_db),
):
    """Generate speech with user validation, concurrency control, usage tracking, and performance monitoring"""
    text_length = len(request.text)
    if not current_user.can_use_characters(text_length):
        raise HTTPException(
            status_code=403,
            detail="Monthly character limit exceeded"
        )

    # Concurrency control: check if user has a pending or processing generation
    existing_job = db.query(GenerationQueue).filter(
        GenerationQueue.user_id == current_user.id,
        GenerationQueue.status.in_(["pending", "processing"])
    ).first()
    if existing_job:
        raise HTTPException(
            status_code=403,
            detail="Only one generation request allowed at a time per user."
        )

    # Add job to generation queue
    job = GenerationQueue(
        user_id=current_user.id,
        status="pending",
        text_content=request.text[:5000],  # Limit stored text length for queue
        parameters=request.json(),
        created_at=time.time()
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        # Mark job as processing
        job.status = "processing"
        job.started_at = time.time()
        db.commit()

        perf_monitor = utils.PerformanceMonitor(
            enabled=config_manager.get_bool("server.enable_performance_monitor", False)
        )
        perf_monitor.record("TTS request received")

        if not engine.MODEL_LOADED:
            logger.error("TTS request failed: Model not loaded.")
            raise HTTPException(
                status_code=503,
                detail="TTS engine model is not currently loaded or available.",
            )

        audio_prompt_path_for_engine: Optional[Path] = None
        if request.voice_mode == "predefined":
            if not request.predefined_voice_id:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'predefined_voice_id' for 'predefined' voice mode.",
                )
            voices_dir = get_predefined_voices_path(ensure_absolute=True)
            potential_path = voices_dir / request.predefined_voice_id
            if not potential_path.is_file():
                logger.error(f"Predefined voice file not found: {potential_path}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Predefined voice file '{request.predefined_voice_id}' not found.",
                )
            audio_prompt_path_for_engine = potential_path
            logger.info(f"Using predefined voice: {request.predefined_voice_id}")

        elif request.voice_mode == "clone":
            if not request.reference_audio_filename:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'reference_audio_filename' for 'clone' voice mode.",
                )
            ref_dir = get_reference_audio_path(ensure_absolute=True)
            potential_path = ref_dir / request.reference_audio_filename
            if not potential_path.is_file():
                logger.error(f"Reference audio file for cloning not found: {potential_path}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Reference audio file '{request.reference_audio_filename}' not found.",
                )
            max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
            is_valid, msg = utils.validate_reference_audio(potential_path, max_dur)
            if not is_valid:
                raise HTTPException(
                    status_code=400, detail=f"Invalid reference audio: {msg}"
                )
            audio_prompt_path_for_engine = potential_path
            logger.info(f"Using reference audio for cloning: {request.reference_audio_filename}")

        perf_monitor.record("Parameters and voice path resolved")

        all_audio_segments_np: List[np.ndarray] = []
        final_output_sample_rate = get_audio_sample_rate()
        engine_output_sample_rate: Optional[int] = None

        if request.split_text and len(request.text) > (request.chunk_size * 1.5 if request.chunk_size else 120 * 1.5):
            chunk_size_to_use = request.chunk_size if request.chunk_size is not None else 120
            logger.info(f"Splitting text into chunks of size ~{chunk_size_to_use}.")
            text_chunks = utils.chunk_text_by_sentences(request.text, chunk_size_to_use)
            perf_monitor.record(f"Text split into {len(text_chunks)} chunks")
        else:
            text_chunks = [request.text]
            logger.info("Processing text as a single chunk (splitting not enabled or text too short).")

        if not text_chunks:
            raise HTTPException(
                status_code=400, detail="Text processing resulted in no usable chunks."
            )

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
            try:
                chunk_audio_tensor, chunk_sr_from_engine = engine.synthesize(
                    text=chunk,
                    audio_prompt_path=str(audio_prompt_path_for_engine) if audio_prompt_path_for_engine else None,
                    temperature=request.temperature if request.temperature is not None else get_gen_default_temperature(),
                    exaggeration=request.exaggeration if request.exaggeration is not None else get_gen_default_exaggeration(),
                    cfg_weight=request.cfg_weight if request.cfg_weight is not None else get_gen_default_cfg_weight(),
                    seed=request.seed if request.seed is not None else get_gen_default_seed(),
                )
                perf_monitor.record(f"Engine synthesized chunk {i+1}")

                if chunk_audio_tensor is None or chunk_sr_from_engine is None:
                    error_detail = f"TTS engine failed to synthesize audio for chunk {i+1}."
                    logger.error(error_detail)
                    raise HTTPException(status_code=500, detail=error_detail)

                if engine_output_sample_rate is None:
                    engine_output_sample_rate = chunk_sr_from_engine
                elif engine_output_sample_rate != chunk_sr_from_engine:
                    logger.warning(
                        f"Inconsistent sample rate from engine: chunk {i+1} ({chunk_sr_from_engine}Hz) "
                        f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                    )

                current_processed_audio_tensor = chunk_audio_tensor

                speed_factor_to_use = request.speed_factor if request.speed_factor is not None else get_gen_default_speed_factor()
                if speed_factor_to_use != 1.0:
                    current_processed_audio_tensor, _ = utils.apply_speed_factor(
                        current_processed_audio_tensor,
                        chunk_sr_from_engine,
                        speed_factor_to_use,
                    )
                    perf_monitor.record(f"Speed factor applied to chunk {i+1}")

                processed_audio_np = current_processed_audio_tensor.cpu().numpy().squeeze()
                all_audio_segments_np.append(processed_audio_np)

            except HTTPException as http_exc:
                raise http_exc
            except Exception as e_chunk:
                error_detail = f"Error processing audio chunk {i+1}: {str(e_chunk)}"
                logger.error(error_detail, exc_info=True)
                raise HTTPException(status_code=500, detail=error_detail)

        if not all_audio_segments_np:
            logger.error("No audio segments were successfully generated.")
            raise HTTPException(
                status_code=500, detail="Audio generation resulted in no output."
            )

        if engine_output_sample_rate is None:
            logger.error("Engine output sample rate could not be determined.")
            raise HTTPException(
                status_code=500, detail="Failed to determine engine sample rate."
            )

        try:
            final_audio_np = (
                np.concatenate(all_audio_segments_np)
                if len(all_audio_segments_np) > 1
                else all_audio_segments_np[0]
            )
            perf_monitor.record("All audio chunks processed and concatenated")

            if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
                final_audio_np = utils.trim_lead_trail_silence(final_audio_np, engine_output_sample_rate)
                perf_monitor.record(f"Global silence trim applied")

            if config_manager.get_bool("audio_processing.enable_internal_silence_fix", False):
                final_audio_np = utils.fix_internal_silence(final_audio_np, engine_output_sample_rate)
                perf_monitor.record(f"Global internal silence fix applied")

            if config_manager.get_bool("audio_processing.enable_unvoiced_removal", False) and utils.PARSELMOUTH_AVAILABLE:
                final_audio_np = utils.remove_long_unvoiced_segments(final_audio_np, engine_output_sample_rate)
                perf_monitor.record(f"Global unvoiced removal applied")

        except ValueError as e_concat:
            logger.error(f"Audio concatenation failed: {e_concat}", exc_info=True)
            for idx, seg in enumerate(all_audio_segments_np):
                logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
            raise HTTPException(status_code=500, detail=f"Audio concatenation error: {e_concat}")

        output_format_str = request.output_format if request.output_format else get_audio_output_format()

        encoded_audio_bytes = utils.encode_audio(
            audio_array=final_audio_np,
            sample_rate=engine_output_sample_rate,
            output_format=output_format_str,
            target_sample_rate=final_output_sample_rate,
        )
        perf_monitor.record(f"Final audio encoded to {output_format_str} (target SR: {final_output_sample_rate}Hz from engine SR: {engine_output_sample_rate}Hz)")

        if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
            logger.error(f"Failed to encode final audio to format: {output_format_str} or output is too small ({len(encoded_audio_bytes or b'')} bytes).")
            raise HTTPException(status_code=500, detail=f"Failed to encode audio to {output_format_str} or generated invalid audio.")

        media_type = f"audio/{output_format_str}"
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        suggested_filename_base = f"tts_output_{timestamp_str}"
        download_filename = utils.sanitize_filename(f"{suggested_filename_base}.{output_format_str}")
        headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

        logger.info(f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}.")
        logger.debug(perf_monitor.report())

        # Mark job as completed
        job.status = "completed"
        job.completed_at = time.time()
        db.commit()

        # Track usage
        generation_time = time.time() - job.started_at
        usage_log = UsageLog(
            user_id=current_user.id,
            characters_used=text_length,
            text_content=request.text[:500],  # Store first 500 chars
            voice_mode=request.voice_mode,
            voice_file=request.predefined_voice_id or request.reference_audio_filename,
            generation_time=generation_time
        )
        db.add(usage_log)

        # Update user's character usage
        current_user.use_characters(text_length)
        db.commit()

        return StreamingResponse(io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers)

    except Exception as e:
        logger.error(f"TTS Generation error: {e}", exc_info=True)
        # Mark job as failed
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = time.time()
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

# --- API Endpoint for Initial UI Data ---
@app.get("/api/ui/initial-data", tags=["UI Helpers"])
async def get_ui_initial_data(
    current_user: User = Depends(auth_handler.get_current_user)
):
    """Provides all necessary initial data for the UI to render"""
    logger.info("Request received for /api/ui/initial-data.")
    try:
        full_config = get_full_config_for_template()
        reference_files = utils.get_valid_reference_files()
        predefined_voices = utils.get_predefined_voices()
        loaded_presets = []
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
                else:
                    logger.warning(f"Invalid format in {presets_file}. Expected a list.")

        initial_gen_result_placeholder = {
            "outputUrl": None,
            "filename": None,
            "genTime": None,
            "submittedVoiceMode": None,
            "submittedPredefinedVoice": None,
            "submittedCloneFile": None,
        }

        return {
            "config": full_config,
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load initial data for UI.")

# --- File Upload Endpoints ---
@app.post("/upload_reference", tags=["File Management"])
async def upload_reference_audio_endpoint(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(auth_handler.get_current_user)
):
    """Handles uploading of reference audio files (.wav, .mp3) for voice cloning"""
    logger.info(f"Request to /upload_reference with {len(files)} file(s).")
    ref_path = get_reference_audio_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append({"filename": "Unknown", "error": "File received with no filename."})
            logger.warning("Upload attempt with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = ref_path / safe_filename

        try:
            if not (safe_filename.lower().endswith(".wav") or safe_filename.lower().endswith(".mp3")):
                raise ValueError("Invalid file type. Only .wav and .mp3 are allowed.")

            if destination_path.exists():
                logger.info(f"Reference file '{safe_filename}' already exists. Skipping duplicate upload.")
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Successfully saved uploaded reference file to: {destination_path}")

            max_duration = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
            is_valid, validation_msg = utils.validate_reference_audio(destination_path, max_duration)
            if not is_valid:
                logger.warning(f"Uploaded file '{safe_filename}' failed validation: {validation_msg}. Deleting.")
                destination_path.unlink(missing_ok=True)
                upload_errors.append({"filename": safe_filename, "error": validation_msg})
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_reference_files = utils.get_valid_reference_files()
    response_data = {
        "message": f"Processed {len(files)} file(s).",
        "uploaded_files": uploaded_filenames_successfully,
        "all_reference_files": all_current_reference_files,
        "errors": upload_errors,
    }
    status_code = 200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    if upload_errors:
        logger.warning(f"Upload to /upload_reference completed with {len(upload_errors)} error(s).")
    return JSONResponse(content=response_data, status_code=status_code)

@app.post("/upload_predefined_voice", tags=["File Management"])
async def upload_predefined_voice_endpoint(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(auth_handler.get_current_admin_user)  # Only admin can upload predefined voices
):
    """Handles uploading of predefined voice files (.wav, .mp3)"""
    logger.info(f"Request to /upload_predefined_voice with {len(files)} file(s).")
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append({"filename": "Unknown", "error": "File received with no filename."})
            logger.warning("Upload attempt for predefined voice with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = predefined_voices_path / safe_filename

        try:
            if not (safe_filename.lower().endswith(".wav") or safe_filename.lower().endswith(".mp3")):
                raise ValueError("Invalid file type. Only .wav and .mp3 are allowed for predefined voices.")

            if destination_path.exists():
                logger.info(f"Predefined voice file '{safe_filename}' already exists. Skipping duplicate upload.")
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Successfully saved uploaded predefined voice file to: {destination_path}")

            is_valid, validation_msg = utils.validate_reference_audio(destination_path, max_duration_sec=None)
            if not is_valid:
                logger.warning(f"Uploaded predefined voice '{safe_filename}' failed validation: {validation_msg}. Deleting.")
                destination_path.unlink(missing_ok=True)
                upload_errors.append({"filename": safe_filename, "error": validation_msg})
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing predefined voice file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_predefined_voices = utils.get_predefined_voices()
    response_data = {
        "message": f"Processed {len(files)} predefined voice file(s).",
        "uploaded_files": uploaded_filenames_successfully,
        "all_predefined_voices": all_current_predefined_voices,
        "errors": upload_errors,
    }
    status_code = 200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    if upload_errors:
        logger.warning(f"Upload to /upload_predefined_voice completed with {len(upload_errors)} error(s).")
    return JSONResponse(content=response_data, status_code=status_code)

# --- Configuration Management API Endpoints ---
@app.post("/save_settings", response_model=UpdateStatusResponse, tags=["Configuration"])
async def save_settings_endpoint(
    request: Request,
    current_user: User = Depends(auth_handler.get_current_admin_user)  # Only admin can change settings
):
    """Saves partial configuration updates to the config.yaml file."""
    logger.info("Request received for /save_settings.")
    try:
        partial_update = await request.json()
        if not isinstance(partial_update, dict):
            raise ValueError("Request body must be a JSON object for /save_settings.")
        logger.debug(f"Received partial config data to save: {partial_update}")

        if config_manager.update_and_save(partial_update):
            restart_needed = any(
                key in partial_update
                for key in ["server", "tts_engine", "paths", "model"]
            )
            message = "Settings saved successfully."
            if restart_needed:
                message += " A server restart may be required for some changes to take full effect."
            return UpdateStatusResponse(message=message, restart_needed=restart_needed)
        else:
            logger.error("Failed to save configuration via config_manager.update_and_save.")
            raise HTTPException(
                status_code=500,
                detail="Failed to save configuration file due to an internal error.",
            )
    except ValueError as ve:
        logger.error(f"Invalid data format for /save_settings: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing /save_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings save: {str(e)}",
        )

@app.post("/reset_settings", response_model=UpdateStatusResponse, tags=["Configuration"])
async def reset_settings_endpoint(
    current_user: User = Depends(auth_handler.get_current_admin_user)  # Only admin can reset settings
):
    """Resets the configuration in config.yaml back to hardcoded defaults."""
    logger.warning("Request received to reset all configurations to default values.")
    try:
        if config_manager.reset_and_save():
            logger.info("Configuration successfully reset to defaults and saved.")
            return UpdateStatusResponse(
                message="Configuration reset to defaults. Please reload the page. A server restart may be beneficial.",
                restart_needed=True,
            )
        else:
            logger.error("Failed to reset and save configuration via config_manager.")
            raise HTTPException(
                status_code=500, detail="Failed to reset and save configuration file."
            )
    except Exception as e:
        logger.error(f"Error processing /reset_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings reset: {str(e)}",
        )

@app.post("/restart_server", response_model=UpdateStatusResponse, tags=["Configuration"])
async def restart_server_endpoint(
    current_user: User = Depends(auth_handler.get_current_admin_user)  # Only admin can restart server
):
    """Attempts to trigger a server restart."""
    logger.info("Request received for /restart_server.")
    message = (
        "Server restart initiated. If running locally without a process manager, "
        "you may need to restart manually. For managed environments (Docker, systemd), "
        "the manager should handle the restart."
    )
    logger.warning(message)
    return UpdateStatusResponse(message=message, restart_needed=True)

# --- OpenAI Compatible Endpoint ---
@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(
    request: OpenAISpeechRequest,
    current_user: User = Depends(auth_handler.get_current_user)
):
    """OpenAI-compatible speech generation endpoint."""
    # Check character limit
    text_length = len(request.input_)
    if not current_user.can_use_characters(text_length):
        raise HTTPException(
            status_code=403,
            detail="Monthly character limit exceeded"
        )

    # Determine the audio prompt path based on the voice parameter
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    reference_audio_path = get_reference_audio_path(ensure_absolute=True)
    voice_path_predefined = predefined_voices_path / request.voice
    voice_path_reference = reference_audio_path / request.voice

    if voice_path_predefined.is_file():
        audio_prompt_path = voice_path_predefined
    elif voice_path_reference.is_file():
        audio_prompt_path = voice_path_reference
    else:
        raise HTTPException(
            status_code=404, detail=f"Voice file '{request.voice}' not found."
        )

    # Check if the TTS model is loaded
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    try:
        # Use the provided seed or the default
        seed_to_use = request.seed if request.seed is not None else get_gen_default_seed()

        # Synthesize the audio
        audio_tensor, sr = engine.synthesize(
            text=request.input_,
            audio_prompt_path=str(audio_prompt_path),
            temperature=get_gen_default_temperature(),
            exaggeration=get_gen_default_exaggeration(),
            cfg_weight=get_gen_default_cfg_weight(),
            seed=seed_to_use,
        )

        if audio_tensor is None or sr is None:
            raise HTTPException(
                status_code=500, detail="TTS engine failed to synthesize audio."
            )

        # Apply speed factor if not 1.0
        if request.speed != 1.0:
            audio_tensor, _ = utils.apply_speed_factor(audio_tensor, sr, request.speed)

        # Convert tensor to numpy array
        audio_np = audio_tensor.cpu().numpy()

        # Ensure it's 1D
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze()

        # Encode the audio to the requested format
        encoded_audio = utils.encode_audio(
            audio_array=audio_np,
            sample_rate=sr,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
        )

        if encoded_audio is None:
            raise HTTPException(status_code=500, detail="Failed to encode audio.")

        # Track usage
        generation_time = time.time()
        usage_log = UsageLog(
            user_id=current_user.id,
            characters_used=text_length,
            text_content=request.input_[:500],
            voice_mode="openai",
            voice_file=request.voice,
            generation_time=generation_time
        )
        db = next(get_db())
        db.add(usage_log)
        
        # Update user's character usage
        current_user.use_characters(text_length)
        db.commit()

        # Determine the media type and return the streaming response
        media_type = f"audio/{request.response_format}"
        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except Exception as e:
        logger.error(f"Error in openai_speech_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- UI Helper API Endpoints ---
@app.get("/get_reference_files", response_model=List[str], tags=["UI Helpers"])
async def get_reference_files_api(
    current_user: User = Depends(auth_handler.get_current_user)
):
    """Returns a list of valid reference audio filenames"""
    logger.debug("Request for /get_reference_files.")
    try:
        return utils.get_valid_reference_files()
    except Exception as e:
        logger.error(f"Error getting reference files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve reference audio files.")

@app.get("/get_predefined_voices", response_model=List[Dict[str, str]], tags=["UI Helpers"])
async def get_predefined_voices_api(
    current_user: User = Depends(auth_handler.get_current_user)
):
    """Returns a list of predefined voices with display names and filenames"""
    logger.debug("Request for /get_predefined_voices.")
    try:
        return utils.get_predefined_voices()
    except Exception as e:
        logger.error(f"Error getting predefined voices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve predefined voices list.")

if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()
    
    logger.info(f"Starting VoiceAI Server on http://{server_host}:{server_port}")
    logger.info(f"API documentation will be available at http://{server_host}:{server_port}/docs")
    logger.info(f"Web UI will be available at http://{server_host}:{server_port}/")
    
    uvicorn.run(
        "server_new:app",     # <-- note module name matches this file
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False
    )
