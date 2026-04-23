"""
FastAPI Backend for Sentiment Analysis Model
Loads pre-trained SVM+GloVe model and exposes prediction endpoint
"""

import pickle
import os
import logging
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for sentiment prediction"""
    text: str


class PredictionResponse(BaseModel):
    """Response model for sentiment prediction"""
    sentiment: str
    confidence: float
    label_id: int


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    message: str


# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="Sentiment Analysis API",
    description="Real-time sentiment classification using SVM+GloVe model",
    version="1.0.0"
)

# ============================================================================
# Model Loading
# ============================================================================

MODEL_PACKAGE = None
MODEL_LOADED = False
ERROR_MESSAGE = ""


def find_latest_model():
    """Find the latest .pkl model file in current directory"""
    pkl_files = glob.glob('best_model_svm_glove_*.pkl')
    if not pkl_files:
        return None
    # Return the most recently modified file
    return max(pkl_files, key=os.path.getctime)


def load_model():
    """Load the model package from pickle file"""
    global MODEL_PACKAGE, MODEL_LOADED, ERROR_MESSAGE
    
    try:
        # Find latest model file
        model_file = find_latest_model()
        
        if not model_file:
            ERROR_MESSAGE = "No model file found. Expected: best_model_svm_glove_*.pkl"
            logger.error(ERROR_MESSAGE)
            MODEL_LOADED = False
            return False
        
        logger.info(f"Loading model from: {model_file}")
        
        with open(model_file, 'rb') as f:
            MODEL_PACKAGE = pickle.load(f)
        
        # Validate package structure
        required_keys = ['model', 'glove_embeddings', 'reverse_mapping', 'label_mapping']
        missing_keys = [k for k in required_keys if k not in MODEL_PACKAGE]
        
        if missing_keys:
            ERROR_MESSAGE = f"Model package missing keys: {missing_keys}"
            logger.error(ERROR_MESSAGE)
            MODEL_LOADED = False
            return False
        
        MODEL_LOADED = True
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Classes: {list(MODEL_PACKAGE['reverse_mapping'].values())}")
        logger.info(f"  GloVe embeddings: {len(MODEL_PACKAGE['glove_embeddings'])} words")
        
        return True
    
    except Exception as e:
        ERROR_MESSAGE = f"Failed to load model: {str(e)}"
        logger.error(ERROR_MESSAGE)
        MODEL_LOADED = False
        return False


# ============================================================================
# Helper Functions
# ============================================================================

def text_to_glove_embedding(text: str, glove_embeddings: dict, embedding_dim: int = 100) -> np.ndarray:
    """
    Convert text to GloVe embedding by averaging word vectors
    
    Args:
        text: Input text string
        glove_embeddings: Dictionary of word -> embedding vectors
        embedding_dim: Dimension of embeddings
    
    Returns:
        Averaged embedding vector
    """
    if not text or not text.strip():
        return np.zeros(embedding_dim)
    
    words = text.lower().split()
    valid_vectors = [
        glove_embeddings[word] 
        for word in words 
        if word in glove_embeddings
    ]
    
    if valid_vectors:
        return np.mean(valid_vectors, axis=0)
    else:
        # Return zero vector if no words found in vocab
        logger.warning(f"No words found in vocabulary for text: {text[:50]}...")
        return np.zeros(embedding_dim)


def get_confidence(prediction: np.ndarray) -> float:
    """
    Extract confidence score from model prediction
    
    Priority:
    1. Use predict_proba() if available
    2. Use decision_function() normalized via softmax
    3. Fall back to 1.0 (no probability available)
    """
    model = MODEL_PACKAGE['model']
    
    try:
        # Try predict_proba first (works for some SVM configurations)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([prediction])[0]
            confidence = float(np.max(proba))
            logger.debug(f"Confidence from predict_proba: {confidence:.4f}")
            return confidence
    except Exception as e:
        logger.debug(f"predict_proba failed: {e}")
    
    try:
        # Try decision_function with softmax normalization
        if hasattr(model, 'decision_function'):
            decision = model.decision_function([prediction])[0]
            # Softmax normalization
            exp_decision = np.exp(decision - np.max(decision))
            proba = exp_decision / np.sum(exp_decision)
            confidence = float(np.max(proba))
            logger.debug(f"Confidence from decision_function+softmax: {confidence:.4f}")
            return confidence
    except Exception as e:
        logger.debug(f"decision_function failed: {e}")
    
    # Fall back to 1.0
    logger.debug("Using default confidence: 1.0")
    return 1.0


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    logger.info("Starting Sentiment Analysis API...")
    load_model()
    if MODEL_LOADED:
        logger.info("✓ API ready to serve requests")
    else:
        logger.error(f"✗ API startup failed: {ERROR_MESSAGE}")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse with status and model loading info
    """
    if MODEL_LOADED:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            message="Sentiment Analysis API is running"
        )
    else:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            message=f"Model failed to load: {ERROR_MESSAGE}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for input text
    
    Args:
        request: PredictionRequest with 'text' field
    
    Returns:
        PredictionResponse with sentiment, confidence, and label_id
    
    Raises:
        HTTPException: If model not loaded or text is empty
    """
    # Validate model is loaded
    if not MODEL_LOADED:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {ERROR_MESSAGE}"
        )
    
    # Validate input text
    text = request.text.strip()
    if not text:
        logger.warning("Empty text provided for prediction")
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    try:
        logger.info(f"Predicting sentiment for: {text[:50]}...")
        
        # Convert text to embedding
        embedding = text_to_glove_embedding(
            text,
            MODEL_PACKAGE['glove_embeddings'],
            embedding_dim=100
        )
        
        # Get prediction
        model = MODEL_PACKAGE['model']
        label_id = int(model.predict([embedding])[0])
        sentiment = MODEL_PACKAGE['reverse_mapping'][label_id]
        
        # Get confidence score
        confidence = get_confidence(embedding)
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        logger.info(f"✓ Prediction: {sentiment} (confidence: {confidence:.4f})")
        
        return PredictionResponse(
            sentiment=sentiment,
            confidence=confidence,
            label_id=label_id
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Sentiment Analysis API Server...")
    logger.info("API will be available at http://localhost:8000")
    logger.info("API docs available at http://localhost:8000/docs")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
