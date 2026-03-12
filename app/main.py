"""
Emotion Detection Voice API
FastAPI application for detecting emotions from audio files.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from typing import Optional
import logging

from app.models.emotion_detector import EmotionDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Detection Voice API",
    description="API for detecting emotions from audio files using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize emotion detector model
emotion_detector = None


@app.on_event("startup")
async def startup_event():
    """Initialize the emotion detection model on startup."""
    global emotion_detector
    try:
        emotion_detector = EmotionDetector()
        logger.info("Emotion detector model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load emotion detector model: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global emotion_detector
    emotion_detector = None
    logger.info("Emotion detector model unloaded")


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Emotion Detection Voice API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "predict": "/predict-emotion",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if emotion_detector is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict-emotion")
async def predict_emotion(audio_file: UploadFile = File(...)):
    """
    Predict emotion from audio file.
    
    Args:
        audio_file: Audio file in WAV, MP3, or similar format
    
    Returns:
        JSON response with emotion and confidence score
    """
    if emotion_detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file extension
    allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_path = tmp_file.name
            content = await audio_file.read()
            tmp_file.write(content)
        
        # Perform emotion detection
        result = emotion_detector.predict(tmp_path)
        
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "emotion": result["emotion"],
                "confidence": float(result["confidence"]),
                "all_emotions": result["all_emotions"]
            }
        )
    
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        
        logger.error(f"Error during emotion prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
