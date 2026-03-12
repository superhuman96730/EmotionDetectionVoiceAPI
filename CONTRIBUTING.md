# Contributing Guidelines

## Getting Started

### Prerequisites
- Python 3.9+
- pip or conda
- Git
- Docker (optional)

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/emotion-detection-api.git
cd EmotionDetectionVoiceAPI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Development Workflow

### Branching Strategy
- Use `main` for production-ready code
- Create feature branches: `feature/your-feature-name`
- Create bugfix branches: `bugfix/issue-description`
- Use descriptive branch names

### Commit Guidelines
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `style:`
- Keep commits small and focused
- Write clear, descriptive commit messages
- Example: `feat: Add emotion confidence threshold configuration`

### Testing
- Write tests for new features
- Run tests before submitting PR:
```bash
pytest tests/
pytest --cov=app tests/
```

### Code Style
- Follow PEP 8
- Use type hints where possible
- Document public functions with docstrings
- Use meaningful variable names

### Pull Request Process
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Write/update tests
5. Submit a pull request with clear description
6. Ensure all CI checks pass
7. Request review from maintainers

### Reporting Issues
- Use GitHub Issues
- Provide clear title and description
- Include steps to reproduce
- Attach logs or screenshots if relevant
- Specify your environment (OS, Python version, etc.)

## Code of Conduct
Be respectful and professional in all interactions.
