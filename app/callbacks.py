import numpy as np
from tensorflow.keras.callbacks import Callback

class SafeProgressCallback(Callback):
    def __init__(self):
        super().__init__()
        self.progress = 0.0  # Native float
    
    def on_predict_batch_end(self, batch, logs=None):
        self.progress = float(batch / self.params['steps'])
    
    def get_progress(self):
        return self.progress