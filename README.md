# PocketTTS API Server

A FastAPI-based Text-to-Speech server using PocketTTS with support for built-in voices and voice cloning.

## Features

- **Built-in Voices**: 8 pre-trained voices (alba, marius, javert, jean, fantine, cosette, eponine, azelma)
- **Voice Cloning**: Clone voices from uploaded audio samples
- **Multiple Audio Formats**: Output in WAV, MP3, or OGG format
- **Flexible Return Types**: Stream audio files directly or save to server and return URL
- **Long Script Support**: Automatically chunks long scripts into manageable segments
- **Optimized Performance**: Caches voice states and models for faster generation
- **Cross-platform**: Works on Windows, Linux, and macOS

## Installation

### Prerequisites

- Python 3.12 (recommended). Avoid Python 3.13 to prevent audio library issues.
- FFmpeg (required for MP3/OGG conversion)

#### Install FFmpeg

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Linux (Fedora):**
```bash
sudo dnf install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from `https://ffmpeg.org/download.html` and add to PATH

### Python Dependencies

1. Clone or download this repository

2. Create a Conda environment with Python 3.12 (recommended):
```bash
conda create -n pocket-tts python=3.12
```

3. Activate the Conda environment:
```bash
conda activate pocket-tts
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### 1. Create `.env` file

Create a `.env` file in the project root with the following:

```env
API_KEY=your_secret_api_key_here
PORT=8000
MAX_REF_DURATION=15
MAX_CHARS_PER_CHUNK=218
```

- `API_KEY`: Your secret API key for authenticating requests (required)
- `PORT`: Server port (defaults to 8000 if not specified)
- `MAX_REF_DURATION`: Maximum duration in seconds for reference audio trimming in voice cloning (defaults to 15)
- `MAX_CHARS_PER_CHUNK`: Maximum characters per chunk when splitting long scripts (defaults to 218)

### 2. HuggingFace Authentication (for Voice Cloning)

Voice cloning requires access to the PocketTTS model on HuggingFace.

1. **Accept Terms**: Visit `https://huggingface.co/kyutai/pocket-tts` and accept the terms

2. **Get Your Token**: Go to `https://huggingface.co/settings/tokens` and create a token

3. **Login with Hugging Face CLI:**
```bash
hf auth login
```
Enter your Hugging Face token (from step 2) when prompted.

The token will be saved automatically and used for model downloads.

## Running the Server

### Start the Server

```bash
python server.py
```

The server will:
- Load the TTS model (may take a few minutes on first run as models download)
- Download models to `./models` directory
- Start listening on `0.0.0.0:8000` (accessible from network)

### Server Endpoints

#### Health Check
```bash
GET http://localhost:8000/health
```

#### Root (API Info)
```bash
GET http://localhost:8000/
```

## API Usage

### 1. Text-to-Speech (Built-in Voices)

**Endpoint:** `POST /tts`

**Headers:**
- `API_KEY`: Your API key from `.env`

**Request Body (JSON):**
```json
{
  "script": "Hello, this is a test of the text-to-speech system.",
  "model": "alba",
  "type": "file",
  "format": "mp3"
}
```

**Parameters:**
- `script` (required): Text to convert to speech
- `model` (optional, default: "alba"): Voice model name  
  - Available: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`
- `type` (optional, default: "file"): Return type  
  - `file`: Stream audio file directly  
  - `url`: Save to server and return URL
- `format` (optional, default: "mp3"): Audio format  
  - `wav`: WAV format  
  - `mp3`: MP3 format (compressed)  
  - `ogg`: OGG format (compressed)

**Example - Stream MP3:**
```bash
curl -X POST http://localhost:8000/tts \
  -H "API_KEY: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"script": "Hello world", "format": "mp3", "type": "file"}' \
  --output speech.mp3
```

**Example - Get URL:**
```bash
curl -X POST http://localhost:8000/tts \
  -H "API_KEY: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"script": "Hello world", "format": "mp3", "type": "url"}'
```

**Response (type="url"):**
```json
{
  "url": "http://192.168.1.100:8000/voice/audio_1234567890_abc12345.mp3",
  "filename": "audio_1234567890_abc12345.mp3"
}
```

### 2. Voice Cloning

**Endpoint:** `POST /tts/clone`

**Headers:**
- `API_KEY`: Your API key from `.env`

**Form Data:**
- `script` (required): Text to convert to speech
- `voice_file` (required): Audio file to clone voice from (WAV, MP3, etc.)
- `type` (optional, default: "file"): Return type (`file` or `url`)
- `format` (optional, default: "mp3"): Audio format (`wav`, `mp3`, or `ogg`)

**Example - Clone Voice:**
```bash
curl -X POST http://localhost:8000/tts/clone \
  -H "API_KEY: your_api_key" \
  -F "script=Hello, this is a cloned voice" \
  -F "voice_file=@sample.wav" \
  -F "format=mp3" \
  -F "type=file" \
  --output cloned_speech.mp3
```

**Example - Get URL:**
```bash
curl -X POST http://localhost:8000/tts/clone \
  -H "API_KEY: your_api_key" \
  -F "script=Hello, this is a cloned voice" \
  -F "voice_file=@sample.wav" \
  -F "format=mp3" \
  -F "type=url"
```

**Note:** 
- Reference audio is trimmed to the maximum duration specified in `MAX_REF_DURATION` (default: 15 seconds)
- The first portion of reference audio (up to max duration) is used for voice cloning
- Generated audio automatically removes the reference audio from the output

## How It Works

### Architecture

1. **Model Loading**: On startup, the TTS model loads into memory (cached for performance)
2. **Voice State Caching**: Built-in voices and cloned voices are cached to avoid reloading
3. **Script Chunking**: Long scripts (exceeding MAX_CHARS_PER_CHUNK) are automatically split into chunks
4. **Audio Generation**: Each chunk is generated separately and concatenated
5. **Format Conversion**: Audio is converted to requested format (WAV/MP3/OGG) using pydub
6. **Return Options**: 
   - `type="file"`: Streams audio directly
   - `type="url"`: Saves to `./voice` folder and returns accessible URL

### Directory Structure

```text
PocketTTS/
├── server.py              # Main server file
├── requirements.txt       # Python dependencies
├── .env                   # Configuration (create this)
├── models/                # Downloaded models (auto-created)
├── voice/                 # Saved audio files (auto-created)
└── README.md              # This file
```

### Model Storage

- Models download to `./models` directory (not `~/.cache`)
- HuggingFace cache uses `./models` directory
- Authentication token stays in default location (`~/.huggingface/token`)

### Performance Optimizations

- Voice states are cached in memory (no reloading between requests)
- Cloned voices are cached by file hash (same file = instant reuse)
- Long scripts are chunked to prevent memory issues
- MP3 encoding is optimized (no unnecessary conversions)

## Troubleshooting

### Models Not Downloading

- Ensure you're logged into HuggingFace: `hf auth login`
- Check that you've accepted terms at `https://huggingface.co/kyutai/pocket-tts`
- Verify internet connection and HuggingFace access

### FFmpeg Not Found

- Install FFmpeg (see Prerequisites)
- Ensure FFmpeg is in your system PATH
- Restart the server after installing FFmpeg

### Port Already in Use

- Change `PORT` in `.env` file
- Or kill the process using the port:
  ```bash
  # Linux/macOS
  lsof -ti:8000 | xargs kill
  
  # Windows
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  ```

### Authentication Errors

- Verify `API_KEY` in `.env` matches the header value
- Check that `.env` file is in the project root
- Ensure no extra spaces in `.env` file

## API Response Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## License

This project uses PocketTTS. Please refer to PocketTTS license terms.

## Support

For issues related to:
- **PocketTTS**: `https://github.com/kyutai-labs/pocket-tts`
- **HuggingFace**: `https://huggingface.co/kyutai/pocket-tts`

