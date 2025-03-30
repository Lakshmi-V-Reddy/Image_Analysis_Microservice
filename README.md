# Industrial Equipment Analysis System

A Streamlit-based dashboard for detecting equipment defects and safety compliance in industrial settings using computer vision and deep learning.

## Features

- **Defect Detection**: CNN model identifies faulty equipment
- **Safety Gear Detection**: YOLO model detects PPE (helmets, gloves, etc.)
- **Interactive Dashboard**: User-friendly interface with real-time results
- **Configurable Settings**: Adjustable confidence thresholds and processing modes
- **Multi-format Support**: Works with JPG, PNG, and other common image formats

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/industrial-equipment-analysis.git
cd industrial-equipment-analysis
```
2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

3. Install dependencies: 
pip install -r requirements.txt

4. Configuration
Modify app/config.py to adjust:

-Model paths

-Input image sizes

-Detection thresholds

-Output settings

6. Requirements

-Python 3.8+

-TensorFlow 2.x

-PyTorch

-OpenCV

-Streamlit

-Ultralytics (for YOLO)

License
MIT License - See LICENSE for details