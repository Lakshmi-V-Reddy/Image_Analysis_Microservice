# setup.py
from setuptools import setup, find_packages

setup(
    name="industrial_equipment_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.12.0',
        'opencv-python',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'ultralytics'
    ],
)