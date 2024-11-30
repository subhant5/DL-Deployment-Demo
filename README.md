# Image Classification API

## Overview
A production-grade image classification service using TensorFlow's MobileNetV2 pre-trained model on ImageNet dataset.

## Features
- FastAPI backend
- TensorFlow MobileNetV2 model
- Docker containerization
- Health check endpoint
- Robust error handling
- Logging support

## Prerequisites
- Docker
- Docker Compose
- Python 3.9+

## Installation

### Local Development
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

### Docker Deployment
```bash
# Build and run the container
docker-compose up --build

# Stop the container
docker-compose down
```

## API Endpoints
- `/predict` (POST): Upload an image for classification
- `/health` (GET): Health check endpoint

## Example Usage
```python
import requests

url = 'http://localhost:8000/predict'
files = {'file': open('image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## Production Considerations
- Use HTTPS in production
- Implement authentication
- Configure logging and monitoring
- Use a production WSGI server like Gunicorn

## License
MIT License
