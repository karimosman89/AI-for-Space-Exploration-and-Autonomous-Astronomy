# 🚀 Getting Started Guide

Welcome to AI for Space Exploration! This guide will help you get up and running quickly.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Git**
- **8GB+ RAM** (16GB recommended)
- **NVIDIA GPU** (optional, but recommended for training)

## 🔧 Installation

### Option 1: Quick Start (Recommended)

```bash
# Clone the repository
git clone https://github.com/karimosman89/AI-for-Space-Exploration-and-Autonomous-Astronomy.git
cd AI-for-Space-Exploration-and-Autonomous-Astronomy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Download pre-trained models
python scripts/download_models.py
```

### Option 2: Docker Installation

```bash
# Build Docker image
docker build -t space-ai:latest .

# Run container with API
docker run -p 8000:8000 space-ai:latest

# Or use Docker Compose
docker-compose up
```

## 🎯 Quick Examples

### 1. Classify Satellite Image

```python
from src.models import SatelliteImageClassifier
from src.utils import load_image

# Initialize model
model = SatelliteImageClassifier()

# Load and classify image
image = load_image('path/to/satellite_image.jpg')
result = model.predict(image)

print(f"Terrain: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 2. Detect Astronomical Objects

```python
from src.models import AstronomicalDetector

# Initialize detector
detector = AstronomicalDetector()

# Detect objects
image = load_image('path/to/telescope_image.jpg')
detections = detector.detect(image)

for obj in detections:
    print(f"Found {obj['class']} at ({obj['bbox']['center_x']}, {obj['bbox']['center_y']})")
```

### 3. Plan Spacecraft Trajectory

```python
from src.models import NavigationAgent
import numpy as np

# Initialize agent
agent = NavigationAgent()

# Define start and goal
start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])  # [pos, vel, fuel]
goal = np.array([10, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Plan trajectory
actions, states, reward = agent.plan_trajectory(start, goal)
print(f"Trajectory planned with {len(actions)} steps")
print(f"Total reward: {reward:.2f}")
```

## 🌐 Web Interface

### Launch Streamlit Dashboard

```bash
# Start the dashboard
streamlit run app/dashboard.py

# Access at http://localhost:8501
```

### Launch FastAPI Server

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Access API docs at http://localhost:8000/docs
```

## 📊 Using the API

### Example API Requests

```bash
# Health check
curl http://localhost:8000/health

# Classify satellite image
curl -X POST "http://localhost:8000/api/v1/satellite/classify" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@satellite_image.jpg"

# Plan trajectory
curl -X POST "http://localhost:8000/api/v1/navigation/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "start_state": [0,0,0,0,0,0,0,0,0,1,0,0],
    "goal_state": [10,5,3,0,0,0,0,0,0,0,0,0],
    "max_steps": 1000
  }'
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## 📚 Training Models

### Train Satellite Classifier

```bash
python scripts/train_model.py \
  --model satellite_classifier \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001
```

### Train with Custom Config

```bash
python scripts/train_model.py \
  --model object_detector \
  --config config/detector_config.yaml
```

## 🐛 Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
```bash
# Reduce batch size
python scripts/train_model.py --batch-size 16

# Or use CPU
python scripts/train_model.py --device cpu
```

**Issue: Module not found**
```bash
# Reinstall package
pip install -e .
```

**Issue: Port already in use**
```bash
# Use different port
uvicorn src.api.main:app --port 8001
```

## 📖 Next Steps

1. **Explore Notebooks**: Check out `notebooks/` for interactive tutorials
2. **Read Documentation**: Browse `docs/` for detailed guides
3. **Try Examples**: Run scripts in `examples/` directory
4. **Customize Models**: Modify models in `src/models/`
5. **Contribute**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

## 🆘 Getting Help

- **Documentation**: Read the full docs in `docs/`
- **Issues**: Report bugs on [GitHub Issues](https://github.com/karimosman89/AI-for-Space-Exploration-and-Autonomous-Astronomy/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/karimosman89/AI-for-Space-Exploration-and-Autonomous-Astronomy/discussions)
- **Email**: Contact karim.osman@example.com

## 🎓 Learning Resources

- **Jupyter Notebooks**: Interactive tutorials in `notebooks/`
- **API Documentation**: Available at `/docs` endpoint
- **Example Projects**: See `examples/` directory
- **Blog Posts**: Coming soon!

---

**Ready to explore space with AI? Let's go! 🚀**
