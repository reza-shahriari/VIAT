# VIAT - Video-Image Annotation Tool

VIAT is a desktop application for annotating videos or Image with bounding boxes and custom attributes. It provides an intuitive interface for creating, editing, and managing annotations for computer vision datasets.

## Features

- **Video Playback Controls**: Play, pause, navigate between frames, and use a slider to move through the video
- **Bounding Box Annotation**: Create and edit bounding boxes with customizable class labels
- **Class Management**: Define and customize annotation classes with color coding
- **Attribute Support**: Add custom attributes to annotations
- **Auto-save**: Automatic saving of projects to prevent data loss
- **Keyboard Shortcuts**: Efficient workflow with keyboard navigation

## Installation

### Prerequisites
- Python 3.6+
- PyQt5
- OpenCV (cv2)
- NumPy

### Setup

1. Clone the repository:
```bash
[git clone https://github.com/yourusername/viat.git](https://github.com/reza-shahriari/VIAT.git)
cd viat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python -m run.py
```

## Usage

### Loading a Video
1. Go to File → Open Video
2. Select a video file to annotate

### Creating Annotations
1. Select a class from the class selector
2. Click and drag on the video frame to create a bounding box
3. Edit the bounding box properties in the annotation panel

### Navigation
- Use the playback controls to navigate through the video
- Press Space to play/pause the video
- Use the frame slider to jump to a specific frame

### Saving Your Work
- Projects are automatically saved to the `autosave` directory
- Use File → Save Project to save your work to a specific location

## Keyboard Shortcuts

- **Space**: Play/Pause video
- **Left Arrow**: Previous frame
- **Right Arrow**: Next frame
- **Delete**: Remove selected annotation

## Project File Format

VAT uses a JSON-based file format to store annotations and class definitions. The format includes:
- Annotation data (position, size, class, attributes)
- Class definitions with color information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

Copyright (c) 2023 VAT Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
