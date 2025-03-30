import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # CNN Model Configuration
    CNN_INPUT_SIZE = (128, 128)
    CNN_MODEL_PATH = os.getenv('CNN_MODEL_PATH', 'models/equipment_defect.h5')
    
    # YOLO Configuration
    YOLO_MODEL = os.getenv('YOLO_MODEL', 'yolov5su.pt')
    YOLO_CLASSES = ['helmet', 'gloves', 'goggles', 'vest']
    
    # Dataset Paths
    DATA_PATH = os.getenv('DATA_PATH', 'data')
    DEFECTED_PATH = os.path.join(DATA_PATH, 'Defected')
    NON_DEFECTED_PATH = os.path.join(DATA_PATH, 'Non-Defected')