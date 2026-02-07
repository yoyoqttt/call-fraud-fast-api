from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
from datetime import datetime
import json

# Import our custom modules
from audio_processor import AudioProcessor
from fraud_detector import FraudDetector

# Initialize FastAPI app
app = FastAPI(title="Fraud Call Detection API", version="1.0.0")


# Simple middleware to log all incoming requests (method/path/origin) to help debug CORS/preflight issues
@app.middleware("http")
async def log_requests(request: Request, call_next):
    origin = request.headers.get("origin")
    print(f"[REQ] {request.method} {request.url.path} origin={origin} content-type={request.headers.get('content-type')}")
    return await call_next(request)

# Configure CORS to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our processors
audio_processor = AudioProcessor()
fraud_detector = FraudDetector()

# Directory paths
AUDIO_DIR = "./audio"
TRANSCRIPT_DIR = "./transcripts"
HISTORY_DIR = "./history"

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Fraud Call Detection API is running",
        "version": "1.0.0",
        "endpoints": [
            "/upload-audio",
            "/process-live-recording",
            "/train-model",
            "/get-history",
            "/clear-history",
            "/health"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_trained": fraud_detector.is_model_trained(),
        "transcription_available": getattr(audio_processor, "available", False)
    }


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    
    try:
        # Validate file type
        allowed_extensions = ['.mp3', '.wav', '.flac', '.m4a']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uploaded_{timestamp}{file_extension}"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File saved: {filename}")
        
        # Convert audio to text
        print("Converting speech to text...")

        # If transcription is not available, skip transcription and return a clear response
        if not getattr(audio_processor, "available", False):
            note = "Transcription not available. Install 'transformers' and 'torch' to enable transcription."
            print(f"Warning: {note}")

            result = {
                "success": True,
                "filename": filename,
                "timestamp": timestamp,
                "transcription_available": False,
                "transcript": "",
                "fraud_detection": None,
                "note": note
            }

            # Save to history as an entry without transcript
            save_to_history(result)

            print("Upload saved without transcription")
            return JSONResponse(content=result)

        # Otherwise perform transcription (handle failures gracefully)
        try:
            transcript = audio_processor.transcribe_audio(file_path)
            transcription_success = True
            transcription_error = None
        except HTTPException as he:
            # Re-raise HTTPException so the outer handler does not convert to 500
            raise he
        except Exception as e:
            # Log traceback server-side for debugging and return a structured response
            import traceback
            traceback.print_exc()
            transcription_success = False
            transcription_error = str(e)
            transcript = ""

        # If transcription succeeded, save transcript and run fraud detection
        if transcription_success:
            transcript_filename = f"{timestamp}_transcript.txt"
            transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_filename)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            print(f"Transcript saved: {transcript_filename}")

            # Detect fraud
            print("Analyzing for fraud...")
            fraud_result = fraud_detector.predict(transcript)
            fraud_detection = {
                "is_fraud": fraud_result["is_fraud"],
                "confidence": fraud_result["confidence"],
                "risk_level": fraud_result["risk_level"],
                "fraud_indicators": fraud_result["fraud_indicators"]
            }
        else:
            fraud_detection = None

        # Prepare response
        result = {
            "success": True,
            "filename": filename,
            "timestamp": timestamp,
            "transcription_available": True,
            "transcription_success": transcription_success,
            "transcription_error": transcription_error,
            "transcript": transcript,
            "fraud_detection": fraud_detection
        }
        
        # Save to history
        save_to_history(result)
        
        print("Analysis complete!")
        return JSONResponse(content=result)       
        
    except HTTPException as he:
        # Propagate HTTPExceptions (like 400) unchanged
        raise he
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.options("/process-live-recording")
async def process_live_recording_options(request: Request):
    """Explicit OPTIONS handler to assist with CORS/preflight from browsers."""
    print(f"[OPTIONS] Received preflight for /process-live-recording origin={request.headers.get('origin')}")
    return JSONResponse(content={"ok": True})


@app.post("/process-live-recording")
async def process_live_recording(request: Request, file: UploadFile = File(...)):
    """
    Endpoint 2: Process live recording from microphone
    (Improved logging: logs request method and content-type for debugging CORS/preflight issues)
    
    Steps:
    1. Save recording as MP3
    2. Convert to text
    3. Detect fraud
    4. Save to history
    5. Return results
    """
    try:
        # Log request details for debugging
        try:
            method = request.method
            content_type = request.headers.get('content-type')
        except Exception:
            method = None
            content_type = None
        print(f"[LIVE] Incoming request method={method}, content-type={content_type}")

        # Generate filename for live recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"live_recording_{timestamp}.mp3"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        # Save the recording
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Live recording saved: {filename}")
        
        # Convert to text
        print("Converting speech to text...")

        # If transcription is not available, skip transcription and return a clear response
        if not getattr(audio_processor, "available", False):
            note = "Transcription not available. Install 'transformers' and 'torch' to enable transcription."
            print(f"Warning: {note}")

            result = {
                "success": True,
                "filename": filename,
                "timestamp": timestamp,
                "recording_type": "live",
                "transcription_available": False,
                "transcript": "",
                "fraud_detection": None,
                "note": note
            }

            save_to_history(result)
            print("Live upload saved without transcription")
            return JSONResponse(content=result)

        # Otherwise perform transcription (handle failures gracefully)
        try:
            transcript = audio_processor.transcribe_audio(file_path)
            transcription_success = True
            transcription_error = None
        except HTTPException as he:
            raise he
        except Exception as e:
            import traceback
            traceback.print_exc()
            transcription_success = False
            transcription_error = str(e)
            transcript = ""

        # If transcription succeeded, save transcript and run fraud detection
        if transcription_success:
            transcript_filename = f"{timestamp}_transcript.txt"
            transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_filename)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            print(f"Transcript saved: {transcript_filename}")

            # Detect fraud
            print("Analyzing for fraud...")
            fraud_result = fraud_detector.predict(transcript)
            fraud_detection = {
                "is_fraud": fraud_result["is_fraud"],
                "confidence": fraud_result["confidence"],
                "risk_level": fraud_result["risk_level"],
                "fraud_indicators": fraud_result["fraud_indicators"]
            }
        else:
            fraud_detection = None

        # Prepare response
        result = {
            "success": True,
            "filename": filename,
            "timestamp": timestamp,
            "recording_type": "live",
            "transcription_available": True,
            "transcription_success": transcription_success,
            "transcription_error": transcription_error,
            "transcript": transcript,
            "fraud_detection": fraud_detection
        }
        
        # Save to history
        save_to_history(result)
        
        print("Live recording analysis complete!")
        return JSONResponse(content=result)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train-model")
async def train_model():
    """
    Endpoint 3: Train the fraud detection model
    
    Uses existing transcripts to train/retrain the model
    """
    try:
        print("Starting model training...")
        
        # Train the model
        metrics = fraud_detector.train_model()
        
        print("Model training complete!")
        
        return {
            "success": True,
            "message": "Model trained successfully",
            "metrics": metrics
        }
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-history")
async def get_history():
    """
    Endpoint 4: Retrieve analysis history
    
    Returns all previous fraud detection results
    """
    try:
        history_file = os.path.join(HISTORY_DIR, "detection_history.json")
        
        if not os.path.exists(history_file):
            return {"history": []}
        
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        return {"history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear-history")
async def clear_history():
    """
    Endpoint 5: Clear all history
    
    Removes all history records
    """
    try:
        history_file = os.path.join(HISTORY_DIR, "detection_history.json")
        
        if os.path.exists(history_file):
            os.remove(history_file)
        
        return {
            "success": True,
            "message": "History cleared successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """
    Endpoint 6: Get system statistics
    """
    try:
        # Count files
        audio_files = len([f for f in os.listdir(AUDIO_DIR) if f.endswith(('.mp3', '.wav', '.flac', '.m4a'))])
        transcript_files = len([f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith('.txt')])
        
        # Get history count
        history_file = os.path.join(HISTORY_DIR, "detection_history.json")
        history_count = 0
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
                history_count = len(history)
        
        return {
            "audio_files": audio_files,
            "transcripts": transcript_files,
            "history_records": history_count,
            "model_trained": fraud_detector.is_model_trained()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def save_to_history(result):
    """
    Helper function to save detection results to history
    """
    history_file = os.path.join(HISTORY_DIR, "detection_history.json")
    
    # Load existing history
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new result
    history.append(result)
    
    # Keep only last 100 records
    if len(history) > 100:
        history = history[-100:]
    
    # Save updated history
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Fraud Call Detection API Server")
    print("=" * 60)
    print(f"Audio Directory: {os.path.abspath(AUDIO_DIR)}")
    print(f"Transcript Directory: {os.path.abspath(TRANSCRIPT_DIR)}")
    print(f"History Directory: {os.path.abspath(HISTORY_DIR)}")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")