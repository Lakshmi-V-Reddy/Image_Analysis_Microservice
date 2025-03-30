import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras import callbacks as kcallbacks  # Renamed import
import os
# app/cnn_model.py
from .config import Config  # Use relative import

class EquipmentDefectModel:
    def __init__(self):
        self.config = Config()
        self.model = self.build_model()
        
    def build_model(self):
        """Build CNN model with transfer learning"""
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
    
    
    def train(self, train_images, train_labels, val_images, val_labels, epochs=10):
        """Train the model"""
        cb = [  # Changed variable name from callbacks to cb
            kcallbacks.EarlyStopping(patience=3, restore_best_weights=True),
            kcallbacks.ModelCheckpoint(
                self.config.CNN_MODEL_PATH,
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        history = self.model.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=epochs,
            batch_size=32,
            callbacks=cb  # Updated variable name
        )
        
        return history
    
    def load_weights(self):
        """Load pre-trained weights"""
        if os.path.exists(self.config.CNN_MODEL_PATH):
            self.model.load_weights(self.config.CNN_MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model weights not found at {self.config.CNN_MODEL_PATH}")
    
    def predict(self, image):
        """Make prediction on a single image"""
        return self.model.predict(image)