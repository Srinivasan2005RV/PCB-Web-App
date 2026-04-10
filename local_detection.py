"""
Local PCB Detection Script
Run detection locally on Windows, upload results to MongoDB
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from datetime import datetime
from pymongo import MongoClient
import json
import argparse

# MongoDB Connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://pcbuser:pcbbatch18@cluster0.bhdtezw.mongodb.net/?appName=Cluster0')
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = client['pcb_detection']
detections_collection = db['detections']

# Model path
MODEL_PATH = str(Path(__file__).parent / "Web_service" / "runs" / "pcb_detect" / "pcb_custom_v1_continued" / "weights" / "best.pt")

def load_model():
    """Load YOLO model"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return None
    
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    return model

def run_detection_local(image_path, conf_threshold=0.15, iou_threshold=0.45, user_id="local_user"):
    """
    Run detection locally and push to MongoDB
    
    Args:
        image_path: Path to image file
        conf_threshold: Detection confidence threshold
        iou_threshold: IoU threshold for NMS
        user_id: User identifier for tracking
    
    Returns:
        Detection result dict
    """
    model = load_model()
    if model is None:
        return None
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"\n📸 Processing: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ Cannot read image file")
        return None
    
    print(f"   Size: {img.shape}")
    
    # Run detection
    print(f"🔍 Running detection (conf={conf_threshold}, iou={iou_threshold})")
    results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
    
    # Extract detections
    detections = []
    defect_types = {}
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        print(f"✅ Found {len(boxes)} defects:")
        
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            class_name = model.names.get(class_id, f"class_{class_id}")
            confidence = float(boxes.conf[i].item())
            
            print(f"   - {class_name}: {confidence:.2f}")
            
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
            defect_types[class_name] = defect_types.get(class_name, 0) + 1
    else:
        print(f"✅ No defects detected")
    
    # Get annotated image
    annotated_img = results[0].plot()
    
    # Convert to base64
    _, img_buffer = cv2.imencode('.jpg', img)
    original_base64 = base64.b64encode(img_buffer).decode('utf-8')
    
    _, annotated_buffer = cv2.imencode('.jpg', annotated_img)
    annotated_base64 = base64.b64encode(annotated_buffer).decode('utf-8')
    
    # Prepare MongoDB document
    timestamp = datetime.now().isoformat() + "Z"
    doc = {
        "userId": user_id,
        "timestamp": datetime.now(),
        "total_defects": len(detections),
        "detections": detections,
        "defect_summary": defect_types,
        "image_filename": os.path.basename(image_path),
        "image_base64": original_base64,
        "annotated_image_base64": annotated_base64,
        "status": "confirmed",
        "created_at": datetime.now(),
        "device": "local_windows"
    }
    
    # Save to MongoDB
    try:
        result = detections_collection.insert_one(doc)
        detection_id = str(result.inserted_id)
        print(f"\n✅ Saved to MongoDB: {detection_id}")
        print(f"   DB: pcb_detection | Collection: detections")
        return {
            "success": True,
            "detection_id": detection_id,
            "timestamp": timestamp,
            "total_defects": len(detections),
            "defect_types": defect_types
        }
    except Exception as e:
        print(f"❌ MongoDB Error: {e}")
        return None

def process_folder(folder_path, conf_threshold=0.15, user_id="local_user"):
    """
    Process all images in a folder
    """
    print(f"\n📁 Processing folder: {folder_path}")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in folder")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    for image_path in image_files:
        result = run_detection_local(str(image_path), conf_threshold=conf_threshold, user_id=user_id)
        if result:
            print(f"")

def main():
    parser = argparse.ArgumentParser(description="PCB Defect Detection - Local Processing")
    parser.add_argument('image_or_folder', type=str, help='Image file or folder containing images')
    parser.add_argument('--user', type=str, default='local_user', help='User ID for tracking')
    parser.add_argument('--conf', type=float, default=0.15, help='Confidence threshold')
    
    args = parser.parse_args()
    
    path = args.image_or_folder
    
    if os.path.isfile(path):
        # Single image
        run_detection_local(path, conf_threshold=args.conf, user_id=args.user)
    elif os.path.isdir(path):
        # Folder
        process_folder(path, conf_threshold=args.conf, user_id=args.user)
    else:
        print(f"❌ Path not found: {path}")

if __name__ == "__main__":
    main()
