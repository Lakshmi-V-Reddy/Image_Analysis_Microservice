from ultralytics import YOLO
import numpy as np
from .config import Config

class SafetyGearDetector:
    def __init__(self):
        self.config = Config()
        self.model = YOLO(self.config.YOLO_MODEL)
        
    def detect(self, image):
        """Detect safety gear in image"""
        results = self.model(image)
        detections = []
        
        for result in results:
            for box in result.boxes:
                label_idx = int(box.cls[0])
                if label_idx < len(self.config.YOLO_CLASSES):
                    detections.append({
                        'label': self.config.YOLO_CLASSES[label_idx],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })
        
        return detections