"""
Configuration settings for emotion detection API
"""

from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    api_title: str = "Emotion Detection Voice API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Audio Processing
    sample_rate: int = 22050
    n_mfcc: int = 13
    max_duration: int = 30  # seconds
    
    # Model Settings
    model_path: Optional[str] = None
    confidence_threshold: float = 0.7
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
