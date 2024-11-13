import os
import sys

from gui.app import MainWindow
from models.ink2doodle import Ink2Doodle
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    model = Ink2Doodle("models/ink2doodle.pth")
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec())