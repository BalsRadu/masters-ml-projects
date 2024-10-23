import os
import sys

from gui.app import ImageAugmentationApp
from PyQt6 import QtWidgets

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    app = QtWidgets.QApplication(sys.argv)
    main_window = ImageAugmentationApp()
    main_window.show()
    sys.exit(app.exec())
