import io
import logging
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load pre-trained ImageNet model
try:
    model = tf.keras.applications.MobileNetV2(
        weights='imagenet', 
        include_top=True
    )
    # Preload model to warm up
    dummy_input = tf.zeros((1, 224, 224, 3))
    model.predict(dummy_input)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Load ImageNet class labels
with open(tf.keras.utils.get_file('imagenet_classes.txt', 
          'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'), 'r') as f:
    class_labels = eval(f.read())

app = FastAPI(
    title="ImageNet Classification Service",
    description="A production-grade image classification service using TensorFlow and MobileNetV2",
    version="1.0.0"
)

def preprocess_image(image_file: bytes) -> np.ndarray:
    """
    Preprocess the input image for model prediction.
    
    Args:
        image_file (bytes): Input image bytes
    
    Returns:
        np.ndarray: Preprocessed image tensor
    """
    try:
        # Open image and convert to RGB
        image = Image.open(io.BytesIO(image_file)).convert('RGB')
        
        # Resize image to model's expected input size
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array[np.newaxis, ...]
        )
        
        return image_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.post("/predict", response_model=List[str])
async def predict_image(file: UploadFile = File(...)) -> List[str]:
    """
    Predict image classes using pre-trained MobileNetV2 model.
    
    Args:
        file (UploadFile): Input image file
    
    Returns:
        List[str]: Predicted image classes
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(contents)
        
        # Predict
        predictions = model.predict(processed_image)
        
        # Decode predictions
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
            predictions, top=3
        )[0]
        
        # Extract class names
        results = [f"label name: {label} confidence:({prob:.2f})" for (number, label, prob) in decoded_predictions]
        
        logger.info(f"Prediction results: {results}")
        return results
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service is running.
    
    Returns:
        dict: Health status
    """
    try:
        # Verify model is loaded and can make a prediction
        dummy_input = tf.zeros((1, 224, 224, 3))
        model.predict(dummy_input)
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service is not healthy")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
