# Emotion Detection Voice API - Usage Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

### Local Development
```bash
uvicorn app.main:app --reload
```

### Docker
```bash
docker build -t emotion-detection-api .
docker run -p 8000:8000 emotion-detection-api
```

## API Endpoints

### 1. Root Endpoint
- **URL**: `GET /`
- **Response**: Status message

### 2. Emotion Prediction
- **URL**: `POST /predict`
- **Parameter**: Upload audio file
- **Response**: Detected emotion with confidence scores

### 3. Supported Emotions
- **URL**: `GET /emotions`
- **Response**: List of supported emotion categories

## Example Usage

```python
import requests

# Predict emotion from audio file
with open('sample.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())

# Get supported emotions
emotions = requests.get('http://localhost:8000/emotions')
print(emotions.json())
```

## Supported Audio Formats
- WAV
- MP3
- FLAC
- OGG

## Configuration
See `app/config.py` for configurable parameters like sample rate, MFCC coefficients, and model path.
