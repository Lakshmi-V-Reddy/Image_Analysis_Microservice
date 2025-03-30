import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train import train_model

if __name__ == "__main__":
    train_model()