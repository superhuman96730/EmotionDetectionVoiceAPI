"""
Integration tests for emotion detection API
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.emotion_detector import EmotionDetector
import numpy as np

client = TestClient(app)

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_emotions_endpoint(self):
        """Test emotions endpoint"""
        response = client.get("/emotions")
        assert response.status_code == 200
        data = response.json()
        assert "emotions" in data
        assert len(data["emotions"]) > 0
    
    def test_predict_endpoint_no_file(self):
        """Test predict endpoint without file"""
        response = client.post("/predict")
        assert response.status_code == 422
    
    def test_predict_endpoint_with_file(self):
        """Test predict endpoint with file"""
        # Create dummy audio file
        audio_data = b"test_audio_data"
        response = client.post(
            "/predict",
            files={"file": ("test.wav", audio_data)}
        )
        # Should return 200 or 400 depending on audio validation
        assert response.status_code in [200, 400, 422]

class TestEmotionDetector:
    """Integration tests for emotion detector"""
    
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        detector = EmotionDetector()
        assert detector is not None
        assert len(detector.EMOTIONS) > 0
    
    def test_emotions_list_complete(self):
        """Test all expected emotions are present"""
        detector = EmotionDetector()
        expected_emotions = ['happy', 'sad', 'angry', 'neutral', 'surprised', 'disgusted']
        for emotion in expected_emotions:
            assert emotion in detector.EMOTIONS
