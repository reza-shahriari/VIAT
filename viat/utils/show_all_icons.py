from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QStyle

COL_SIZE = 4


class Widget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Standard Icons')
        layout = QGridLayout(self)
        count = 0
        for attr in dir(QStyle):
            if attr.startswith('SP_'):
                icon_attr = getattr(QStyle, attr)
                btn = QPushButton(attr)
                btn.setIcon(self.style().standardIcon(icon_attr))
                layout.addWidget(btn, count // COL_SIZE, count % COL_SIZE)
                count += 1


if __name__ == '__main__':
    app = QApplication([])
    w = Widget()
    w.show()
    app.exec_()