"""
Data preprocessing module for emotion detection
"""

import numpy as np
from typing import Tuple
import soundfile as sf

class DataPreprocessor:
    """Handle data preprocessing for audio samples"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio
        return audio / max_val
    
    def trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove silence from beginning and end"""
        mask = np.abs(audio) > threshold
        if np.any(mask):
            return audio[mask[0]:-np.argmax(mask[::-1])]
        return audio
    
    def split_into_chunks(self, audio: np.ndarray, chunk_size: int = 22050) -> list:
        """Split audio into overlapping chunks"""
        chunks = []
        for i in range(0, len(audio) - chunk_size, chunk_size // 2):
            chunks.append(audio[i:i + chunk_size])
        return chunks
