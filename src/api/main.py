"""
FastAPI application for AI Space Exploration API.

Provides REST endpoints for:
- Satellite image classification
- Astronomical object detection
- Trajectory planning
- Galaxy classification
- Exoplanet detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from PIL import Image
import io

from src.models import (
    SatelliteImageClassifier,
    AstronomicalDetector,
    NavigationAgent,
    GalaxyClassifier,
    ExoplanetDetector
)

# Initialize FastAPI app
app = FastAPI(
    title="AI Space Exploration API",
    description="Advanced AI/ML API for space exploration and astronomy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (lazy loading in production)
models = {
    'satellite_classifier': None,
    'object_detector': None,
    'navigation_agent': None,
    'galaxy_classifier': None,
    'exoplanet_detector': None
}


def load_model(model_name: str):
    """Lazy load models on first use."""
    if models[model_name] is None:
        if model_name == 'satellite_classifier':
            models[model_name] = SatelliteImageClassifier()
        elif model_name == 'object_detector':
            models[model_name] = AstronomicalDetector()
        elif model_name == 'navigation_agent':
            models[model_name] = NavigationAgent()
        elif model_name == 'galaxy_classifier':
            models[model_name] = GalaxyClassifier()
        elif model_name == 'exoplanet_detector':
            models[model_name] = ExoplanetDetector()
    
    return models[model_name]


# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: List[str]


class TrajectoryRequest(BaseModel):
    start_state: List[float]
    goal_state: List[float]
    max_steps: int = 1000


class LightCurveRequest(BaseModel):
    flux_values: List[float]


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Space Exploration API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "satellite_classify": "/api/v1/satellite/classify",
            "object_detect": "/api/v1/astronomy/detect",
            "plan_trajectory": "/api/v1/navigation/plan",
            "classify_galaxy": "/api/v1/galaxy/classify",
            "detect_exoplanet": "/api/v1/exoplanet/detect"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    loaded_models = [name for name, model in models.items() if model is not None]
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": loaded_models
    }


@app.post("/api/v1/satellite/classify")
async def classify_satellite_image(file: UploadFile = File(...)):
    """
    Classify satellite imagery into terrain types.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Classification results with probabilities
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Load model and predict
        model = load_model('satellite_classifier')
        result = model.predict(image_array)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/astronomy/detect")
async def detect_astronomical_objects(file: UploadFile = File(...)):
    """
    Detect astronomical objects in telescope images.
    
    Args:
        file: Uploaded image file
        
    Returns:
        List of detected objects with bounding boxes
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Load model and detect
        model = load_model('object_detector')
        results = model.detect(image_array)
        
        return JSONResponse(content={"detections": results})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/navigation/plan")
async def plan_trajectory(request: TrajectoryRequest):
    """
    Plan optimal spacecraft trajectory.
    
    Args:
        request: Trajectory planning request with start/goal states
        
    Returns:
        Planned trajectory and total reward
    """
    try:
        # Load model
        agent = load_model('navigation_agent')
        
        # Plan trajectory
        start = np.array(request.start_state)
        goal = np.array(request.goal_state)
        actions, states, reward = agent.plan_trajectory(
            start,
            goal,
            max_steps=request.max_steps
        )
        
        return JSONResponse(content={
            "success": True,
            "num_actions": len(actions),
            "total_reward": float(reward),
            "final_state": [float(x) for x in states[-1][:3]],
            "distance_to_goal": float(np.linalg.norm(states[-1][:3] - goal[:3]))
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/galaxy/classify")
async def classify_galaxy(file: UploadFile = File(...)):
    """
    Classify galaxy morphology.
    
    Args:
        file: Uploaded galaxy image
        
    Returns:
        Galaxy classification result
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Load model and predict
        model = load_model('galaxy_classifier')
        result = model.predict(image_array)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/exoplanet/detect")
async def detect_exoplanet(request: LightCurveRequest):
    """
    Detect exoplanets from light curve data.
    
    Args:
        request: Light curve flux values
        
    Returns:
        Exoplanet detection result
    """
    try:
        # Load model
        model = load_model('exoplanet_detector')
        
        # Detect exoplanet
        light_curve = np.array(request.flux_values)
        result = model.detect(light_curve)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models")
async def list_models():
    """List available models and their status."""
    return {
        "models": [
            {
                "name": "satellite_classifier",
                "description": "Satellite image terrain classification",
                "loaded": models['satellite_classifier'] is not None
            },
            {
                "name": "object_detector",
                "description": "Astronomical object detection",
                "loaded": models['object_detector'] is not None
            },
            {
                "name": "navigation_agent",
                "description": "Autonomous spacecraft navigation",
                "loaded": models['navigation_agent'] is not None
            },
            {
                "name": "galaxy_classifier",
                "description": "Galaxy morphology classification",
                "loaded": models['galaxy_classifier'] is not None
            },
            {
                "name": "exoplanet_detector",
                "description": "Exoplanet detection from light curves",
                "loaded": models['exoplanet_detector'] is not None
            }
        ]
    }


def serve():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    serve()
