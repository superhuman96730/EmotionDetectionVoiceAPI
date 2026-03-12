"""
Emotion Detection Model
Uses PyTorch and Librosa for audio processing and emotion classification.
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Process audio files for emotion detection."""
    
    def __init__(self, sr: int = 22050, n_mfcc: int = 13):
        """
        Initialize audio processor.
        
        Args:
            sr: Sample rate (default: 22050 Hz)
            n_mfcc: Number of MFCC features (default: 13)
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            MFCC features array
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Calculate statistics
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Combine features
            features = np.hstack([mfcc_mean, mfcc_std])
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {str(e)}")
            raise


class EmotionClassifier(nn.Module):
    """PyTorch neural network for emotion classification."""
    
    def __init__(self, input_size: int = 26, num_emotions: int = 4):
        """
        Initialize emotion classifier.
        
        Args:
            input_size: Input feature size (default: 26 - 13 means + 13 stds)
            num_emotions: Number of emotion classes (default: 4)
        """
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, num_emotions)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        x = self.softmax(self.fc4(x))
        return x


class EmotionDetector:
    """Main emotion detection class."""
    
    EMOTIONS = ["happy", "sad", "angry", "neutral"]
    
    def __init__(self, model_path: str = None):
        """
        Initialize emotion detector.
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_processor = AudioProcessor()
        
        # Initialize model
        self.model = EmotionClassifier(input_size=26, num_emotions=len(self.EMOTIONS))
        self.model = self.model.to(self.device)
        
        # Try to load pre-trained model if path provided
        if model_path:
            try:
                self.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {str(e)}")
                logger.info("Using initialized model with random weights")
        else:
            logger.info("Using initialized model (no pre-trained weights)")
        
        self.model.eval()
    
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained model weights.
        
        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        logger.info(f"Model loaded successfully from {model_path}")
    
    def save_model(self, model_path: str) -> None:
        """
        Save model weights.
        
        Args:
            model_path: Path to save model checkpoint
        """
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    def predict(self, audio_path: str) -> Dict:
        """
        Predict emotion from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with predicted emotion and confidence
        """
        try:
            # Extract features
            features = self.audio_processor.extract_features(audio_path)
            
            # Convert to tensor and prepare for model
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(features_tensor)
                probabilities = output[0].cpu().numpy()
            
            # Get emotion with highest confidence
            predicted_idx = np.argmax(probabilities)
            predicted_emotion = self.EMOTIONS[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Build all emotions dict
            all_emotions = {
                emotion: float(prob)
                for emotion, prob in zip(self.EMOTIONS, probabilities)
            }
            
            return {
                "emotion": predicted_emotion,
                "confidence": confidence,
                "all_emotions": all_emotions
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
