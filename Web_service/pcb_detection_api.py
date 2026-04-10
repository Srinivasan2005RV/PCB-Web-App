"""
PCB Defect Detection API - MongoDB Version
FastAPI-based cloud-ready detection service
Stores images and detection history in MongoDB
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import os
import json
from pathlib import Path
import sys
import logging
from bson.objectid import ObjectId

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ MONGODB CONNECTION ============
try:
    from pymongo import MongoClient
    
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Test connection
    client.admin.command('ping')
    
    db = client['pcb_detection']
    detections_collection = db['detections']
    
    # Create indexes for fast queries
    detections_collection.create_index("userId")
    detections_collection.create_index("timestamp")
    
    logger.info("✅ MongoDB connected and ready!")
    MONGODB_ENABLED = True
except Exception as e:
    logger.warning(f"⚠️  MongoDB not available: {e}")
    MONGODB_ENABLED = False
    detections_collection = None

# Load the best performing model
MODEL_PATH = str(Path(__file__).parent / "runs" / "pcb_detect" / "pcb_custom_v1_continued" / "weights" / "best.pt")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {Path(MODEL_PATH).exists()}")
model = None

def load_model():
    """Load the YOLO model"""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise Exception(f"Model not found at {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    return model

# ============ DATABASE FUNCTIONS ============

def save_detection_to_db(detection_data):
    """Save detection result to MongoDB"""
    if not MONGODB_ENABLED or detections_collection is None:
        logger.warning("MongoDB not available, skipping history save")
        return None
    
    try:
        # Prepare document
        doc = {
            "userId": detection_data.get("userId", "mobile_user"),
            "timestamp": datetime.fromisoformat(detection_data.get("timestamp").replace('Z', '+00:00')) if isinstance(detection_data.get("timestamp"), str) else datetime.now(),
            "total_defects": len(detection_data.get("detections", [])),
            "detections": detection_data.get("detections", []),
            "defect_summary": detection_data.get("detection_summary", {}),
            "image_base64": detection_data.get("image_base64"),  # Store image as base64
            "annotated_image_base64": detection_data.get("annotated_image_base64"),
            "status": "confirmed",
            "created_at": datetime.now()
        }
        
        result = detections_collection.insert_one(doc)
        logger.info(f"✅ Detection saved to MongoDB: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {e}")
        return None

def get_detection_history(user_id="mobile_user", limit=50):
    """Get detection history for a user from MongoDB"""
    if not MONGODB_ENABLED or detections_collection is None:
        return []
    
    try:
        items = list(detections_collection.find(
            {"userId": user_id}
        ).sort("timestamp", -1).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for item in items:
            item["_id"] = str(item["_id"])
        
        logger.info(f"Retrieved {len(items)} detections from history")
        return items
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return []

def get_detection_by_id(detection_id):
    """Get a specific detection by ID"""
    if not MONGODB_ENABLED or detections_collection is None:
        return None
    
    try:
        item = detections_collection.find_one({"_id": ObjectId(detection_id)})
        if item:
            item["_id"] = str(item["_id"])
        return item
    except Exception as e:
        logger.error(f"Error retrieving detection: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    try:
        load_model()
        print("🚀 PCB Detection API with MongoDB is ready!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
    yield

# Initialize FastAPI app
app = FastAPI(
    title="PCB Defect Detection API",
    description="AI-powered PCB defect detection service with MongoDB storage",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "PCB Defect Detection API",
        "version": "1.0.0",
        "database": "MongoDB",
        "endpoints": {
            "detect": "/detect - Main detection endpoint",
            "health": "/health - Health check",
            "history": "/history - Get detection history"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    db_connected = MONGODB_ENABLED
    return {
        "status": "healthy" if (model_loaded and db_connected) else "degraded",
        "model_loaded": model_loaded,
        "database_connected": db_connected,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect")
async def detect_defects(
    file: UploadFile = File(...),
    user_id: str = "mobile_user",
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.45
):
    """
    Main detection endpoint - returns JSON with detection results and annotated image
    
    Args:
        file: Image file (JPG, PNG)
        user_id: User identifier for history tracking
        conf_threshold: Confidence threshold (0-1)
        iou_threshold: IoU threshold for NMS (0-1)
    
    Returns:
        JSON with detection results, base64 encoded images, and detection ID
    """
    try:
        print(f"\n📥 Receiving file: {file.filename}")
        print(f"   Content-Type: {file.content_type}")
        
        # Ensure model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Read and validate image
        contents = await file.read()
        print(f"   File size: {len(contents) / 1024:.1f} KB")
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        print(f"   Image shape: {img.shape}")
        
        # Run detection
        print(f"🔍 Running detection (conf={conf_threshold}, iou={iou_threshold})")
        results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
        print(f"✅ Detection complete")
        
        # Extract detection data
        detections = []
        defect_types = {}
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            print(f"   Found {len(boxes)} detections")
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                class_name = model.names.get(class_id, f"class_{class_id}")
                confidence = float(boxes.conf[i].item())
                
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "defect_type": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": float(boxes.xyxy[i][0].item()),
                        "y1": float(boxes.xyxy[i][1].item()),
                        "x2": float(boxes.xyxy[i][2].item()),
                        "y2": float(boxes.xyxy[i][3].item())
                    }
                }
                detections.append(detection)
                
                # Count defect types
                defect_types[class_name] = defect_types.get(class_name, 0) + 1
        else:
            print("   No defects detected")
        
        # Get annotated image (with bounding boxes)
        annotated_img = results[0].plot()
        
        # Convert original image to base64
        _, img_buffer = cv2.imencode('.jpg', img)
        original_base64 = base64.b64encode(img_buffer).decode('utf-8')
        
        # Convert annotated image to base64
        _, annotated_buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(annotated_buffer).decode('utf-8')
        
        timestamp = datetime.now().isoformat() + "Z"
        
        # Prepare response
        response_data = {
            "success": True,
            "timestamp": timestamp,
            "detection_summary": {
                "total_defects": len(detections),
                "defect_types": defect_types
            },
            "detections": detections,
            "image_base64": original_base64,
            "annotated_image_base64": annotated_base64,
            "userId": user_id
        }
        
        # Save to MongoDB
        detection_id = save_detection_to_db(response_data)
        response_data["detection_id"] = detection_id
        
        print(f"Sending response: {response_data['detection_summary']}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/detect/{detection_id}")
async def get_detection(detection_id: str):
    """Get a specific detection result"""
    try:
        result = get_detection_by_id(detection_id)
        if not result:
            raise HTTPException(status_code=404, detail="Detection not found")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(user_id: str = "mobile_user", limit: int = 50):
    """Get detection history for a user"""
    try:
        if limit > 100:
            limit = 100  # Cap at 100
        
        history = get_detection_history(user_id, limit)
        
        return JSONResponse(content={
            "success": True,
            "user_id": user_id,
            "count": len(history),
            "detections": history
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/confirm_defect")
async def confirm_defect(detection_id: str, confirmed: bool = True):
    """Mark a detection as confirmed/reviewed"""
    try:
        if not MONGODB_ENABLED or detections_collection is None:
            raise HTTPException(status_code=500, detail="Database unavailable")
        
        detections_collection.update_one(
            {"_id": ObjectId(detection_id)},
            {"$set": {
                "status": "confirmed" if confirmed else "rejected",
                "reviewed_at": datetime.now()
            }}
        )
        
        return JSONResponse(content={
            "success": True,
            "detection_id": detection_id,
            "status": "confirmed" if confirmed else "rejected"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
