import numpy as np

def debug_float32():
    for name, val in globals().items():
        if isinstance(val, np.float32):
            print(f"Found float32: {name} = {val}")
            globals()[name] = float(val)  # Auto-convert

# Run before your analysis
debug_float32()