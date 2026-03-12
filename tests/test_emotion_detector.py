"""
Unit tests for emotion detector
"""

import pytest
import numpy as np
from app.models.emotion_detector import EmotionDetector
from app.preprocessing import DataPreprocessor

class TestEmotionDetector:
    """Test emotion detector functionality"""
    
    def setup_method(self):
        self.detector = EmotionDetector()
    
    def test_emotions_list(self):
        """Test available emotions"""
        assert len(self.detector.EMOTIONS) == 6
        assert ''happy'' in self.detector.EMOTIONS
        assert ''sad'' in self.detector.EMOTIONS
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.detector.model is None
        assert self.detector.scaler is None

class TestDataPreprocessor:
    """Test data preprocessing"""
    
    def setup_method(self):
        self.preprocessor = DataPreprocessor()
    
    def test_normalize_audio(self):
        """Test audio normalization"""
        audio = np.array([0.5, 1.0, 0.3])
        normalized = self.preprocessor.normalize_audio(audio)
        assert np.max(np.abs(normalized)) <= 1.0
    
    def test_trim_silence(self):
        """Test silence trimming"""
        audio = np.array([0.001, 0.5, 0.8, 0.001])
        trimmed = self.preprocessor.trim_silence(audio)
        assert len(trimmed) < len(audio)
