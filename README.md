


          
# VIAT - Video-Image Annotation Tool

![VIAT Logo](d:/VIAT/viat/Icon/Icon.ico)

VIAT is a powerful annotation tool designed for computer vision tasks, enabling users to create high-quality annotations for both videos and images. With an intuitive interface and comprehensive feature set, VIAT streamlines the annotation process for machine learning and computer vision projects.

## 🚀 Features

### 🎞️ Video Playback Controls
- Smooth video playback with frame-by-frame navigation
- Support for various video formats
- Frame extraction capabilities

### 🖼️ Annotation Capabilities
- **Bounding Box Annotation**: Create precise bounding boxes around objects
- **Multiple Creation Methods**: Support for drag and two-click annotation methods
- **Smart Edge Movement**: Edges snap to image features for precise adjustments
- **Selection and Modification**: Easily select, resize, and modify existing annotations

### 🎨 Class Management
- Support for multiple object classes with customizable colors
- Color-coded annotations for easy identification

### 📝 Attribute Support
- Add custom attributes like Size and Quality to annotations
- Attribute management for consistent labeling

### 💾 Data Management
- **Auto-save Functionality**: Prevents data loss during annotation
- **Project Saving and Loading**: Save your work and continue later
- **Multiple Export Formats**: Export annotations to common formats:
  - COCO JSON
  - YOLO
  - Pascal VOC

### 🔍 View Controls
- Zoom and pan functionality
- Maintain proper aspect ratio for accurate annotation
- Coordinate transformation between display and image space

### 📋 Context Menu
- Right-click context menu for quick editing and deletion of annotations

## 📦 Installation

### Windows Installer

The easiest way to install VIAT is using the Windows installer:

1. Download the latest VIAT_Setup.exe from the releases page
2. Run the installer and follow the on-screen instructions
3. Launch VIAT from the desktop shortcut or start menu

### From Source

To install from source:

```bash
# Clone the repository
git clone https://github.com/yourusername/VIAT.git
cd VIAT

# Option 1: Using pip
pip install -r requirements.txt
python setup.py install

# Option 2: Using conda
conda env create -f environment.yml
conda activate viat
```

## 📂 Project Structure

```
VIAT/
├── viat/                  # Main application source code
│   ├── main.py            # Main application entry point
│   ├── canvas.py          # Video canvas implementation
│   ├── utils/             # Utility functions
│   └── Icon/              # Application icons
├── installation_tools/    # Installation and packaging scripts
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment configuration
├── setup.py               # Installation script
└── README.md              # Project documentation
```

## 💡 Usage

### Getting Started

1. **Open a Video or Image**: 
   - Use File > Open Video/Image or drag and drop files
   - Supported formats include MP4, AVI, JPG, PNG, and more

2. **Create Annotations**:
   - Select an annotation class from the class panel
   - Choose your preferred annotation method (drag or two-click)
   - Draw bounding boxes around objects of interest

3. **Edit Annotations**:
   - Click on a bounding box to select it
   - Drag edges or corners to resize
   - Use Smart Edge Movement for precise adjustments (toggle with the toolbar button)
   - Right-click for additional options

4. **Export Your Work**:
   - Choose File > Export Annotations
   - Select your preferred format (COCO, YOLO, Pascal VOC)
   - Specify the output directory

### Keyboard Shortcuts

- **Space**: Play/Pause video
- **Left/Right Arrow**: Previous/Next frame
- **Ctrl+S**: Save project
- **Ctrl+Z**: Undo
- **Delete**: Remove selected annotation
- **+/-**: Zoom in/out

## 🤝 Contributing

Contributions to VIAT are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact

For questions, feature requests, or bug reports, please open an issue on the GitHub repository.

---

*VIAT - Making annotation easier for computer vision tasks*

        