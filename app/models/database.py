"""
Database models for storing predictions and user data
"""

from datetime import datetime
from typing import Optional

class PredictionRecord:
    """Model for storing prediction records"""
    
    def __init__(self, file_name: str, emotion: str, confidence: float, duration: float):
        self.id: Optional[int] = None
        self.file_name = file_name
        self.emotion = emotion
        self.confidence = confidence
        self.duration = duration
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "file_name": self.file_name,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "duration": self.duration,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class UserSession:
    """Model for tracking user sessions"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.predictions = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def add_prediction(self, prediction):
        """Add prediction to session"""
        self.predictions.append(prediction)
        self.last_activity = datetime.now()
