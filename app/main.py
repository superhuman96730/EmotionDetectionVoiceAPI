"""
FastAPI application for emotion detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from app.models.emotion_detector import EmotionDetector

app = FastAPI(title="Emotion Detection Voice API")
detector = EmotionDetector()

@app.get("/")
def read_root():
    return {"message": "Emotion Detection Voice API"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """Predict emotion from uploaded audio file"""
    try:
        contents = await file.read()
        prediction = detector.predict(contents)
        return {"emotion": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/emotions")
def get_emotions():
    """Get list of supported emotions"""
    return {"emotions": detector.EMOTIONS}
