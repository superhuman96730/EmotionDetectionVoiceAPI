"""
Error handling module for emotion detection API
"""

class EmotionDetectionError(Exception):
    """Base exception for emotion detection errors"""
    pass

class InvalidAudioError(EmotionDetectionError):
    """Raised when audio file is invalid"""
    pass

class ModelNotFoundError(EmotionDetectionError):
    """Raised when model file is not found"""
    pass

class PredictionError(EmotionDetectionError):
    """Raised when prediction fails"""
    pass

class AudioProcessingError(EmotionDetectionError):
    """Raised when audio processing fails"""
    pass
