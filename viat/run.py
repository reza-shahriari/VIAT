#!/usr/bin/env python
import sys
from PyQt5.QtWidgets import QApplication
from main import VideoAnnotationTool


def main():
    app = QApplication(sys.argv)
    window = VideoAnnotationTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
