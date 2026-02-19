import os
import asyncio
import logging
import time
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import assemblyai as aai

# --- 1. Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_debug.log")
    ]
)
logger = logging.getLogger("TranscriptionAPI")

# --- 2. Load Configuration ---
load_dotenv()
ASSEMBLY_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not ASSEMBLY_API_KEY:
    logger.error("ASSEMBLYAI_API_KEY not found in environment variables!")
else:
    aai.settings.api_key = ASSEMBLY_API_KEY
    logger.info("AssemblyAI API Key configured successfully.")

# --- 3. FastAPI App Setup ---
app = FastAPI(
    title="High-Performance Transcription API",
    description="Optimized parallel transcription service."
)

# CORS is already configured - this should work correctly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Middleware for Request Timing ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Duration: {process_time:.2f}s")
    return response

# --- 5. Endpoints ---
@app.post("/transcription")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes audio files with non-blocking I/O and verbose logging.
    """
    request_id = os.urandom(4).hex()
    logger.info(f"[{request_id}] Received transcription request for: {file.filename}")

    if not ASSEMBLY_API_KEY:
        logger.error(f"[{request_id}] API Key missing.")
        raise HTTPException(status_code=500, detail="API Key not configured.")

    try:
        # Step 1: Read file
        logger.info(f"[{request_id}] Reading file into memory...")
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"[{request_id}] File size: {file_size_mb:.2f} MB")

        # Step 2: Configure Transcriber
        transcriber = aai.Transcriber()
        # "best" model provides superior language detection and multilingual support
        config = aai.TranscriptionConfig(
            speech_model="best",
            language_detection=True
        )

        # Step 3: Run Transcription (offloaded to thread pool)
        logger.info(f"[{request_id}] Starting AssemblyAI transcription...")
        loop = asyncio.get_event_loop()
        
        # We use the executor to handle N users without blocking the main event loop
        transcript = await loop.run_in_executor(
            None, 
            lambda: transcriber.transcribe(content, config=config)
        )

        # Step 4: Handle Results
        if transcript.status == aai.TranscriptStatus.error:
            logger.error(f"[{request_id}] AssemblyAI Error: {transcript.error}")
            raise HTTPException(status_code=400, detail=f"AssemblyAI Error: {transcript.error}")

        logger.info(f"[{request_id}] Transcription COMPLETED. Words: {len(transcript.text.split())}")

        return {
            "status": "completed",
            "request_id": request_id,
            "filename": file.filename,
            "transcription": transcript.text,
            "confidence": transcript.confidence
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"[{request_id}] Critical Error: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

@app.get("/")
async def health_check():
    return {"status": "online", "message": "High-Performance Transcription API is ready."}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
