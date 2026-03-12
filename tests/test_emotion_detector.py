"""
Unit tests for emotion detection model
"""

import pytest
import numpy as np
import tempfile
import librosa
from app.models.emotion_detector import (
    AudioProcessor,
    EmotionClassifier,
    EmotionDetector
)


class TestAudioProcessor:
    """Test audio processing functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.processor = AudioProcessor()
    
    def create_test_audio(self, duration=2, sr=22050):
        """Create a temporary test audio file."""
        # Generate simple sine wave
        y = np.sin(2 * np.pi * 440 * np.linspace(0, duration, sr * duration))
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            librosa.output.write_wav(f.name, y, sr)
            return f.name
    
    def test_audio_processor_initialization(self):
        """Test AudioProcessor initialization."""
        assert self.processor.sr == 22050
        assert self.processor.n_mfcc == 13
    
    def test_audio_processor_custom_params(self):
        """Test AudioProcessor with custom parameters."""
        processor = AudioProcessor(sr=16000, n_mfcc=40)
        assert processor.sr == 16000
        assert processor.n_mfcc == 40
    
    def test_extract_features(self):
        """Test feature extraction from audio."""
        audio_path = self.create_test_audio()
        features = self.processor.extract_features(audio_path)
        
        # Features should have shape (26,) = 13 means + 13 stds
        assert features.shape == (26,)
        assert not np.isnan(features).any()


class TestEmotionClassifier:
    """Test emotion classifier model."""
    
    def test_classifier_initialization(self):
        """Test EmotionClassifier initialization."""
        classifier = EmotionClassifier()
        assert classifier is not None
    
    def test_classifier_custom_params(self):
        """Test EmotionClassifier with custom parameters."""
        classifier = EmotionClassifier(input_size=40, num_emotions=6)
        assert classifier is not None
    
    def test_classifier_forward_pass(self):
        """Test forward pass through classifier."""
        import torch
        
        classifier = EmotionClassifier()
        classifier.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 26)
        
        with torch.no_grad():
            output = classifier(dummy_input)
        
        # Check output shape
        assert output.shape == (1, 4)
        
        # Check probabilities sum to 1
        assert np.isclose(output.sum().item(), 1.0, atol=1e-6)
        
        # Check all values are between 0 and 1
        assert (output >= 0).all() and (output <= 1).all()


class TestEmotionDetector:
    """Test emotion detection."""
    
    def setup_method(self):
        """Setup for each test."""
        self.detector = EmotionDetector()
    
    def create_test_audio(self, duration=2, sr=22050):
        """Create a temporary test audio file."""
        # Generate simple sine wave
        y = np.sin(2 * np.pi * 440 * np.linspace(0, duration, sr * duration))
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            librosa.output.write_wav(f.name, y, sr)
            return f.name
    
    def test_detector_initialization(self):
        """Test EmotionDetector initialization."""
        assert self.detector is not None
        assert self.detector.model is not None
    
    def test_emotions_list(self):
        """Test emotions list."""
        expected_emotions = ["happy", "sad", "angry", "neutral"]
        assert self.detector.EMOTIONS == expected_emotions
    
    def test_predict(self):
        """Test prediction on audio file."""
        audio_path = self.create_test_audio()
        result = self.detector.predict(audio_path)
        
        # Check result structure
        assert "emotion" in result
        assert "confidence" in result
        assert "all_emotions" in result
        
        # Check emotion is valid
        assert result["emotion"] in self.detector.EMOTIONS
        
        # Check confidence is between 0 and 1
        assert 0 <= result["confidence"] <= 1
        
        # Check all emotions present
        assert len(result["all_emotions"]) == 4
        
        # Check all confidence scores sum to 1
        total_confidence = sum(result["all_emotions"].values())
        assert np.isclose(total_confidence, 1.0, atol=1e-6)
    
    def test_predict_result_format(self):
        """Test prediction result format."""
        audio_path = self.create_test_audio()
        result = self.detector.predict(audio_path)
        
        # Check all emotions in result
        for emotion in self.detector.EMOTIONS:
            assert emotion in result["all_emotions"]
            assert isinstance(result["all_emotions"][emotion], float)
        
        # Check predicted emotion matches highest confidence
        highest_emotion = max(result["all_emotions"], 
                            key=result["all_emotions"].get)
        assert result["emotion"] == highest_emotion


class TestAPI:
    """Test API endpoints."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test root endpoint."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "Emotion Detection Voice API"
        assert "endpoints" in data
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
