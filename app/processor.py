# app/processor.py
import cv2
import numpy as np
from typing import Union

class ImageProcessor:
    @staticmethod
    def load_image(file_path: Union[str, bytes], target_size=None) -> np.ndarray:
        """Load image with proper type handling"""
        if isinstance(file_path, bytes):
            img = cv2.imdecode(np.frombuffer(file_path, np.uint8), cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(file_path)
        
        if img is None:
            raise ValueError(f"Could not read image from {file_path[:50]}...")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def visualize_detections(image: np.ndarray, detections: list) -> np.ndarray:
        """Draw detection bounding boxes with type safety"""
        img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['label']}: {det['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

    @staticmethod
    def preprocess_for_cnn(image: np.ndarray) -> np.ndarray:
        """Preprocess image with explicit type conversion"""
        processed = cv2.resize(image, (224, 224))
        processed = processed.astype(np.float32) / 255.0  # Model expects float32
        return processed

    @staticmethod
    def convert_to_display_prob(prob: Union[float, np.number]) -> float:
        """Convert numpy float to native Python float for UI components"""
        return float(prob)  # This ensures Streamlit compatibility