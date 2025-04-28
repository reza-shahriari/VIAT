#!/usr/bin/env python
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viat.main import VideoAnnotationTool  

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon




def main():
    app = QApplication(sys.argv)
    window = VideoAnnotationTool()
    window.change_style("DarkModern")

    icon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "Icon", "Icon.png"
    )
    window.setWindowIcon(QIcon(icon_path))
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
