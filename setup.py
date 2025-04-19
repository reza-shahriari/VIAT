from setuptools import setup, find_packages

setup(
    name="vat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyQt5>=5.15.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "pillow>=9.0.0",
        "moviepy>=1.0.0",
        "ffmpeg-python>=0.2.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "filterpy>=1.4.5",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.5.0",
    ],
    entry_points={
        "console_scripts": [
            "vat=vat.run:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Video Annotation Tool with object detection and tracking capabilities",
    keywords="video, annotation, object detection, tracking",
    python_requires=">=3.8",
)
