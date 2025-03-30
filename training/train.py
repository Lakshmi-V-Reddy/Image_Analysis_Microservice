import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cnn_model import EquipmentDefectModel
from app.config import Config

class ImageProcessor:
    @staticmethod
    def load_image(file_path, target_size=(128, 128)):
        """Load and resize an image"""
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image {file_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, target_size)

def load_dataset():
    """Load and prepare dataset"""
    config = Config()
    processor = ImageProcessor()
    
    defected_images = []
    non_defected_images = []
    target_size = config.CNN_INPUT_SIZE
    
    # Load defected images
    defected_dir = config.DEFECTED_PATH
    if not os.path.exists(defected_dir):
        raise FileNotFoundError(f"Defected images directory not found: {defected_dir}")
    
    for img_file in os.listdir(defected_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(defected_dir, img_file)
            try:
                img = processor.load_image(img_path, target_size)
                defected_images.append(img)
            except Exception as e:
                print(f"Skipping {img_path}: {str(e)}")
                continue
    
    # Load non-defected images
    non_defected_dir = config.NON_DEFECTED_PATH
    if not os.path.exists(non_defected_dir):
        raise FileNotFoundError(f"Non-defected images directory not found: {non_defected_dir}")
    
    for img_file in os.listdir(non_defected_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(non_defected_dir, img_file)
            try:
                img = processor.load_image(img_path, target_size)
                non_defected_images.append(img)
            except Exception as e:
                print(f"Skipping {img_path}: {str(e)}")
                continue
    
    # Verify we have images
    if not defected_images or not non_defected_images:
        raise ValueError("No valid images found in dataset directories")
    
    # Create labels (0=defected, 1=non-defected)
    defected_labels = np.zeros(len(defected_images))
    non_defected_labels = np.ones(len(non_defected_images))
    
    # Combine datasets
    images = np.array(defected_images + non_defected_images, dtype=np.float32)
    labels = np.concatenate([defected_labels, non_defected_labels])
    
    # Normalize pixel values
    images = images / 255.0
    
    # Shuffle dataset
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images, labels = images[indices], labels[indices]
    
    return train_test_split(images, labels, test_size=0.2, random_state=42)

def train_model():
    """Main training function"""
    try:
        X_train, X_val, y_train, y_val = load_dataset()
        
        model = EquipmentDefectModel()
        history = model.train(X_train, y_train, X_val, y_val, epochs=15)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
        print(f"Training completed. Model saved to: {Config().CNN_MODEL_PATH}")
        return True
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

if __name__ == '__main__':
    train_model()