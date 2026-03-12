# Emotion Detection Voice API

A machine learning-powered API for detecting emotions from audio files using PyTorch and Librosa. This project demonstrates ML model serving, API deployment, and audio processing.

## Features

- 🎙️ **Audio Upload**: Upload audio files in multiple formats (WAV, MP3, OGG, FLAC, M4A)
- 🧠 **Emotion Detection**: Detect emotions (happy, sad, angry, neutral) from audio
- 📊 **Confidence Scores**: Get confidence scores for each emotion
- 🚀 **Fast & Scalable**: Built with FastAPI for high performance
- 🐳 **Docker Support**: Easy deployment with Docker

## Tech Stack

- **FastAPI**: Modern Python web framework
- **PyTorch**: Deep learning framework for neural networks
- **Librosa**: Audio processing and feature extraction
- **Docker**: Containerization for deployment
- **Uvicorn**: ASGI server for FastAPI

## Project Structure

```
EmotionDetectionVoiceAPI/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   └── models/
│       ├── __init__.py
│       └── emotion_detector.py # Emotion detection logic
├── data/                       # Audio data storage
├── models/                     # Pre-trained model weights
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EmotionDetectionVoiceAPI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Setup

1. **Build the Docker image**
   ```bash
   docker-compose build
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

**Response:**
```json
{
  "name": "Emotion Detection Voice API",
  "version": "1.0.0",
  "status": "online",
  "endpoints": {
    "predict": "/predict-emotion",
    "health": "/health"
  }
}
```

### 2. Health Check
```
GET /health
```
Check if the model is loaded and API is healthy.

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded"
}
```

### 3. Predict Emotion
```
POST /predict-emotion
```
Predict emotion from an uploaded audio file.

**Parameters:**
- `audio_file`: Audio file (WAV, MP3, OGG, FLAC, M4A)

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.87,
  "all_emotions": {
    "happy": 0.87,
    "sad": 0.05,
    "angry": 0.03,
    "neutral": 0.05
  }
}
```

## Usage Examples

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict-emotion" \
  -H "accept: application/json" \
  -F "audio_file=@/path/to/audio.wav"
```

### Using Python Requests

```python
import requests

with open("audio.wav", "rb") as f:
    files = {"audio_file": f}
    response = requests.post("http://localhost:8000/predict-emotion", files=files)
    print(response.json())
```

### Using FastAPI Interactive Docs

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## Model Architecture

The emotion detection model uses a feed-forward neural network with the following architecture:

1. **Audio Processing**:
   - Load audio using Librosa
   - Extract 13 MFCC (Mel-Frequency Cepstral Coefficients) features
   - Calculate mean and standard deviation for each feature

2. **Neural Network**:
   - Input layer: 26 features (13 means + 13 stds)
   - Hidden layer 1: 128 neurons + ReLU + Dropout(0.3)
   - Hidden layer 2: 64 neurons + ReLU + Dropout(0.3)
   - Hidden layer 3: 32 neurons + ReLU + Dropout(0.2)
   - Output layer: 4 neurons (emotions) + Softmax

## Emotion Classes

- **happy**: Positive emotion indicating joy or happiness
- **sad**: Negative emotion indicating sadness or melancholy
- **angry**: Negative emotion indicating anger or frustration
- **neutral**: Neutral emotional state

## Configuration

Environment variables can be set in `.env` file:

```env
PYTHONUNBUFFERED=1
```

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

## Performance Considerations

- **Inference Time**: ~50-200ms per audio file depending on length
- **Memory**: ~500MB with model loaded
- **GPU Support**: Automatically uses CUDA if available
- **Concurrency**: Handles multiple concurrent requests with Uvicorn workers

## Deployment

### Production Deployment with Gunicorn

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

### Cloud Deployment

- **AWS EC2**: Use EC2 instance with Docker
- **AWS ECS**: Use ECS task definition
- **Google Cloud Run**: Deploy containerized application
- **Azure Container Instances**: Run Docker container
- **Heroku**: Deploy using Dockerfile

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

MIT License - See LICENSE file for details

## Future Enhancements

- [ ] Fine-tuning on custom emotion datasets
- [ ] Multi-language support
- [ ] Real-time streaming emotion detection
- [ ] Model version management
- [ ] Advanced audio preprocessing (silence removal, normalization)
- [ ] Batch processing endpoint
- [ ] Web UI for testing
- [ ] Metrics and monitoring dashboard

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Librosa Documentation](https://librosa.org/)
- [MFCC Features](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

## Contact & Support

For issues and questions, please create an issue in the repository.

---

