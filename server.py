import os
import io
import tempfile
import hashlib
import time
import uuid
import socket
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Query, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import scipy.io.wavfile
from pydub import AudioSegment

# Load environment variables from .env file
load_dotenv()

# Set models download directory to ./models in base directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Create voice folder for saved audio files
VOICE_DIR = Path(__file__).parent / "voice"
VOICE_DIR.mkdir(exist_ok=True)

# Get IPv4 address for URL generation
def get_ipv4_address():
    """Get the IPv4 address of the machine."""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to localhost if unable to determine IP
        return "127.0.0.1"

SERVER_IP = get_ipv4_address()
# Get port from environment variable, default to 8000
SERVER_PORT = int(os.getenv("PORT", "8000"))

# Set HuggingFace cache to use ./models directory
# HUGGINGFACE_HUB_CACHE controls where models are cached
# Don't set HF_HOME - that would move the token file, breaking authentication
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR)

# Patch pocket_tts to use our models directory BEFORE importing TTSModel
import pocket_tts.utils.utils as pocket_utils
_original_make_cache_directory = pocket_utils.make_cache_directory
def patched_make_cache_directory():
    """Override to use ./models directory instead of ~/.cache/pocket_tts"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR
pocket_utils.make_cache_directory = patched_make_cache_directory

# Patch hf_hub_download to explicitly use our cache directory
# This ensures all HuggingFace downloads go to ./models
from huggingface_hub import hf_hub_download as _original_hf_hub_download
def patched_hf_hub_download(*args, **kwargs):
    """Override to use ./models directory for HuggingFace downloads"""
    # Always use our models directory, but don't override if explicitly set
    if "cache_dir" not in kwargs:
        kwargs["cache_dir"] = str(MODELS_DIR)
    try:
        # Try to download with our custom cache directory
        result = _original_hf_hub_download(*args, **kwargs)
        return result
    except Exception as e:
        # If download fails, it might be an auth issue - let the error propagate
        # The error message will indicate if it's an authentication problem
        print(f"Warning: HuggingFace download failed: {e}")
        raise
pocket_utils.hf_hub_download = patched_hf_hub_download

# Now import TTSModel after patching
from pocket_tts import TTSModel

# Get API key from environment
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY not found in .env file")

# Available built-in voices (no voice cloning required)
AVAILABLE_VOICES = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']

# Global model and voice states cache (kept in memory for performance)
# load_model() and get_state_for_audio_prompt() are slow, so we cache them
tts_model: TTSModel = None
voice_states: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global tts_model
    print(f"Loading TTS model (models directory: {MODELS_DIR})...")
    print("This may take a few minutes on first run as models download...")
    try:
        tts_model = TTSModel.load_model()
        print("TTS model loaded successfully!")
        print(f"Models are stored in: {MODELS_DIR}")
        print(f"Available voices: {AVAILABLE_VOICES}")
    except Exception as e:
        print(f"Error loading TTS model: {e}")
        print(f"Make sure models can be downloaded to: {MODELS_DIR}")
        raise
    yield
    # Cleanup on shutdown (if needed)
    print("Shutting down TTS server...")


# Initialize FastAPI app
app = FastAPI(
    title="PocketTTS API",
    description="Text-to-Speech API using pocket_tts",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files for voice folder
app.mount("/voice", StaticFiles(directory=str(VOICE_DIR)), name="voice")

# Mount static files for voice folder
app.mount("/voice", StaticFiles(directory=str(VOICE_DIR)), name="voice")


class TTSRequest(BaseModel):
    """Request body for TTS endpoint."""
    script: str = Field(..., description="The text script to convert to speech")
    model: str = Field(default="alba", description="The voice model name. Available: alba, marius, javert, jean, fantine, cosette, eponine, azelma")
    type: str = Field(default="file", description="Return type: file (stream) or url (save and return URL)")
    format: str = Field(default="mp3", description="Output audio format: wav, mp3, or ogg")

def get_voice_state(model_name: str):
    """
    Get or create voice state for a given model/voice name.
    Voice states are cached in memory since get_state_for_audio_prompt() is slow.
    Uses built-in voices that don't require voice cloning access.
    """
    # Validate voice name
    if model_name not in AVAILABLE_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice '{model_name}'. Available voices: {AVAILABLE_VOICES}"
        )
    
    if model_name not in voice_states:
        try:
            print(f"Loading voice state for '{model_name}'...")
            # Use built-in voice from the catalog (no voice cloning required)
            voice_states[model_name] = tts_model.get_state_for_voice(model_name)
            print(f"Voice state for '{model_name}' loaded and cached!")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load voice '{model_name}': {str(e)}"
            )
    return voice_states[model_name]


@app.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    api_key: str = Header(..., alias="API_KEY", description="API key for authentication")
):
    """
    Convert text script to speech and return audio as WAV file.
    
    Request body:
    - **script**: The text script to convert to speech
    - **model**: The voice model name (default: alba-mackenna)
    
    Header:
    - **API_KEY**: API key for authentication
    """
    # Validate API key
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not request.script or not request.script.strip():
        raise HTTPException(status_code=400, detail="Script is required and cannot be empty")
    
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded yet")
    
    try:
        # Get voice state for the requested model (cached for performance)
        voice_state = get_voice_state(request.model)
        
        # Generate audio
        audio = tts_model.generate_audio(voice_state, request.script)
        
        # Validate type and format
        return_type = request.type.lower()
        if return_type not in ["file", "url"]:
            raise HTTPException(status_code=400, detail=f"Unsupported type '{return_type}'. Supported types: file, url")
        
        format = request.format.lower()
        if format not in ["wav", "mp3", "ogg"]:
            raise HTTPException(status_code=400, detail=f"Unsupported format '{format}'. Supported formats: wav, mp3, ogg")
        
        # Convert audio tensor to numpy and prepare for export
        audio_np = audio.numpy()
        # Handle stereo/mono - ensure it's the right shape for scipy
        if len(audio_np.shape) > 1 and audio_np.shape[0] > 1:
            # Multi-channel: transpose to [samples, channels] for scipy
            audio_np = audio_np.T
        elif len(audio_np.shape) > 1:
            # Mono with channel dimension: squeeze it
            audio_np = audio_np[0]
        
        # Write to temporary WAV file first (needed for encoding)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            scipy.io.wavfile.write(tmp_wav.name, tts_model.sample_rate, audio_np)
            tmp_wav_path = tmp_wav.name
        
        try:
            # Convert to requested format using pydub
            audio_segment = AudioSegment.from_wav(tmp_wav_path)
            
            # If URL type, save to voice folder and return URL
            if return_type == "url":
                # Generate unique filename
                unique_id = str(uuid.uuid4())[:8]
                timestamp = int(time.time())
                filename = f"audio_{timestamp}_{unique_id}.{format}"
                file_path = VOICE_DIR / filename
                
                # Export and save to voice folder
                audio_segment.export(str(file_path), format=format, bitrate="128k" if format == "mp3" else None)
                
                # Return URL
                return JSONResponse({
                    "url": f"http://{SERVER_IP}:{SERVER_PORT}/voice/{filename}",
                    "filename": filename
                })
            
            # For file type, stream the audio
            buffer = io.BytesIO()
            if format == "mp3":
                audio_segment.export(buffer, format="mp3", bitrate="128k")
                media_type = "audio/mpeg"
                filename = "speech.mp3"
            elif format == "wav":
                audio_segment.export(buffer, format="wav")
                media_type = "audio/wav"
                filename = "speech.wav"
            elif format == "ogg":
                audio_segment.export(buffer, format="ogg", codec="libvorbis")
                media_type = "audio/ogg"
                filename = "speech.ogg"
            
            buffer.seek(0)
            
            # Return the converted audio
            return StreamingResponse(
                buffer,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


# Cache for cloned voice states (keyed by audio file hash)
cloned_voice_states: dict = {}


@app.post("/tts/clone")
async def text_to_speech_clone(
    script: str = Form(..., description="The text script to convert to speech"),
    voice_file: UploadFile = File(..., description="Audio file to clone voice from (WAV, MP3, etc.)"),
    type: str = Form(default="file", description="Return type: file (stream) or url (save and return URL)"),
    format: str = Form(default="mp3", description="Output audio format: wav, mp3, or ogg"),
    api_key: str = Header(..., alias="API_KEY", description="API key for authentication")
):
    """
    Convert text script to speech using voice cloning from uploaded audio file.
    
    Form data:
    - **script**: The text script to convert to speech
    - **voice_file**: Audio file to clone voice from (WAV format recommended)
    
    Header:
    - **API_KEY**: API key for authentication
    
    Note: Requires access to voice cloning model. Login with `uvx hf auth login` 
    and accept terms at https://huggingface.co/kyutai/pocket-tts
    """
    # Validate API key
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not script or not script.strip():
        raise HTTPException(status_code=400, detail="Script is required and cannot be empty")
    
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded yet")
    
    try:
        # Read the uploaded voice file
        voice_content = await voice_file.read()
        
        if not voice_content:
            raise HTTPException(status_code=400, detail="Voice file is empty")
        
        # Create hash of voice file for caching
        voice_hash = hashlib.md5(voice_content).hexdigest()
        
        # Max duration for reference audio (in seconds) - configurable via env
        MAX_REF_DURATION = float(os.getenv("MAX_REF_DURATION", "15.0"))
        
        if voice_hash not in cloned_voice_states:
            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(voice_content)
                tmp_path = tmp_file.name
            
            try:
                print(f"Loading voice state from uploaded file (hash: {voice_hash[:8]}...)")
                
                # Read the original audio and trim to max duration
                orig_sr, orig_audio = scipy.io.wavfile.read(tmp_path)
                
                # Handle stereo audio - convert to mono
                if len(orig_audio.shape) > 1:
                    orig_audio = orig_audio[:, 0]
                
                # Trim to max duration
                max_samples = int(MAX_REF_DURATION * orig_sr)
                if len(orig_audio) > max_samples:
                    orig_audio = orig_audio[:max_samples]
                    print(f"Trimmed reference audio to {MAX_REF_DURATION}s")
                
                # Save trimmed audio to a new temp file
                trimmed_path = tmp_path + "_trimmed.wav"
                scipy.io.wavfile.write(trimmed_path, orig_sr, orig_audio)
                
                # Load voice state
                voice_state = tts_model.get_state_for_audio_prompt(trimmed_path)
                
                # Generate minimal test audio to measure exact prompt length (cached, so only once per voice)
                test_audio = tts_model.generate_audio(voice_state, ".")
                # Audio tensor shape is [channels, samples] - get samples dimension
                test_samples = test_audio.shape[1] if len(test_audio.shape) > 1 else len(test_audio)
                # The prompt is embedded at the start - measure it by comparing to a known short generation
                # For ".", the actual speech is very short (~0.1s), so most of test_audio is the prompt
                # Use a conservative estimate: test_audio length minus a small buffer for the "." speech
                min_speech_samples = int(0.15 * tts_model.sample_rate)  # ~0.15s buffer for "."
                prompt_samples = max(0, test_samples - min_speech_samples)
                
                # Reload voice state fresh (test generation consumed it)
                voice_state = tts_model.get_state_for_audio_prompt(trimmed_path)
                
                cloned_voice_states[voice_hash] = (voice_state, prompt_samples, trimmed_path)
                print(f"Cloned voice state loaded! (will trim {prompt_samples} samples, ~{prompt_samples/tts_model.sample_rate:.2f}s)")
                
            finally:
                # Clean up original temp file (keep trimmed for reloading voice state)
                os.unlink(tmp_path)
        
        voice_state, prompt_samples, trimmed_path = cloned_voice_states[voice_hash]
        
        # Reload voice state (it gets consumed after generation)
        voice_state = tts_model.get_state_for_audio_prompt(trimmed_path)
        
        # Estimate script duration and split into chunks if needed
        # Max characters per chunk configurable via env (default: 218 = ~17.5s * 12.5 chars/sec)
        target_chars_per_chunk = int(os.getenv("MAX_CHARS_PER_CHUNK", "218"))
        
        # Split script into chunks if it's too long
        script_chunks = []
        if len(script) > target_chars_per_chunk:
            # Split by sentences first, then by length
            sentences = script.replace('!', '.').replace('?', '.').split('.')
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                # Add period back if it was removed
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                
                # If adding this sentence would exceed chunk size, start new chunk
                if current_chunk and len(current_chunk) + len(sentence) + 1 > target_chars_per_chunk:
                    script_chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            if current_chunk:
                script_chunks.append(current_chunk.strip())
        else:
            script_chunks = [script]
        
        # Generate audio for each chunk and concatenate
        audio_chunks = []
        for i, chunk in enumerate(script_chunks):
            # Reload voice state for each chunk (it gets consumed after generation)
            if i > 0:
                voice_state = tts_model.get_state_for_audio_prompt(trimmed_path)
            
            # Generate audio for this chunk
            chunk_audio = tts_model.generate_audio(voice_state, chunk)
            
            # Trim the reference audio from the beginning (only for first chunk)
            if i == 0:
                chunk_audio_samples = chunk_audio.shape[1] if len(chunk_audio.shape) > 1 else len(chunk_audio)
                if chunk_audio_samples > prompt_samples:
                    if len(chunk_audio.shape) > 1:
                        chunk_audio = chunk_audio[:, prompt_samples:]
                    else:
                        chunk_audio = chunk_audio[prompt_samples:]
            
            audio_chunks.append(chunk_audio)
        
        # Concatenate all audio chunks
        import torch
        if len(audio_chunks) > 1:
            # Concatenate along the samples dimension
            if len(audio_chunks[0].shape) > 1:
                audio = torch.cat(audio_chunks, dim=1)  # [channels, samples]
            else:
                audio = torch.cat(audio_chunks, dim=0)  # 1D tensor
        else:
            audio = audio_chunks[0]
        
        # Validate type and format
        return_type = type.lower()
        if return_type not in ["file", "url"]:
            raise HTTPException(status_code=400, detail=f"Unsupported type '{return_type}'. Supported types: file, url")
        
        format = format.lower()
        if format not in ["wav", "mp3", "ogg"]:
            raise HTTPException(status_code=400, detail=f"Unsupported format '{format}'. Supported formats: wav, mp3, ogg")
        
        # Convert audio tensor to numpy and prepare for export
        audio_np = audio.numpy()
        # Handle stereo/mono - ensure it's the right shape for scipy
        if len(audio_np.shape) > 1 and audio_np.shape[0] > 1:
            # Multi-channel: transpose to [samples, channels] for scipy
            audio_np = audio_np.T
        elif len(audio_np.shape) > 1:
            # Mono with channel dimension: squeeze it
            audio_np = audio_np[0]
        
        # Write to temporary WAV file first (needed for encoding)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            scipy.io.wavfile.write(tmp_wav.name, tts_model.sample_rate, audio_np)
            tmp_wav_path = tmp_wav.name
        
        try:
            # Convert to requested format using pydub
            audio_segment = AudioSegment.from_wav(tmp_wav_path)
            
            # If URL type, save to voice folder and return URL
            if return_type == "url":
                # Generate unique filename
                unique_id = str(uuid.uuid4())[:8]
                timestamp = int(time.time())
                filename = f"audio_{timestamp}_{unique_id}.{format}"
                file_path = VOICE_DIR / filename
                
                # Export and save to voice folder
                audio_segment.export(str(file_path), format=format, bitrate="128k" if format == "mp3" else None)
                
                # Return URL
                return JSONResponse({
                    "url": f"http://{SERVER_IP}:{SERVER_PORT}/voice/{filename}",
                    "filename": filename
                })
            
            # For file type, stream the audio
            buffer = io.BytesIO()
            if format == "mp3":
                audio_segment.export(buffer, format="mp3", bitrate="128k")
                media_type = "audio/mpeg"
                filename = "speech.mp3"
            elif format == "wav":
                audio_segment.export(buffer, format="wav")
                media_type = "audio/wav"
                filename = "speech.wav"
            elif format == "ogg":
                audio_segment.export(buffer, format="ogg", codec="libvorbis")
                media_type = "audio/ogg"
                filename = "speech.ogg"
            
            buffer.seek(0)
            
            # Return the converted audio
            return StreamingResponse(
                buffer,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice cloning TTS failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "cached_voices": list(voice_states.keys()),
        "cached_cloned_voices": len(cloned_voice_states),
        "available_voices": AVAILABLE_VOICES
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PocketTTS API",
        "version": "1.0.0",
        "available_voices": AVAILABLE_VOICES,
        "endpoints": {
            "/tts": "POST - Convert text to speech using built-in voices",
            "/tts/clone": "POST - Convert text to speech using voice cloning (upload audio file)",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
