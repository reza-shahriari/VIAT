#!/usr/bin/env python
# Import torch early to avoid DLL initialization conflicts (WinError 1114) with PyQt5 on Windows
try:
    import torch
except ImportError:
    pass

import sys
import os

if getattr(sys, 'frozen', False):
    meipass = sys._MEIPASS
    sys.path.insert(0, meipass)
    sys.path.insert(0, os.path.join(meipass, 'viat'))
else:
    run_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(run_dir))
    sys.path.insert(0, run_dir)

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
