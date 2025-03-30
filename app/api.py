# cnn_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications, layers, models
import os

class EquipmentDefectModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()
        
    def build_model(self):
        """Build ResNet50-based classification model"""
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.CNN_INPUT_SIZE, 3)
        )
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_weights(self):
        """Load pretrained weights"""
        if os.path.exists(self.config.CNN_MODEL_PATH):
            self.model.load_weights(self.config.CNN_MODEL_PATH)
            return True
        return False

    def predict(self, image):
        """Predict defect probability for an image"""
        try:
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            
            # Resize to model's expected input
            image = tf.image.resize(image, self.config.CNN_INPUT_SIZE)
            
            # Normalize and add batch dimension
            image = image.numpy() if hasattr(image, 'numpy') else image
            image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0
            
            # Predict
            pred = self.model.predict(image, verbose=0)
            defect_prob = float(pred[0][1])
            
            return {
                "status": "defect" if defect_prob > 0.5 else "no_defect",
                "confidence": max(defect_prob, 1-defect_prob),
                "probabilities": {
                    "defect": defect_prob,
                    "no_defect": 1-defect_prob
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "confidence": 0.0,
                "probabilities": {
                    "defect": 0.5,
                    "no_defect": 0.5
                }
            }