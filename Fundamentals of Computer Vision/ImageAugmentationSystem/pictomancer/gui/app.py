import os

import cv2
import matplotlib.pyplot as plt
import ruamel.yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout)
from utils.augmentation import apply


class ImageAugmentationApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Augmentation Tool')
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.config_label = QLabel('Select a config file to apply augmentations')
        self.config_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.config_label)

        self.config_button = QPushButton('Load Config File')
        self.config_button.clicked.connect(self.load_config_file)
        layout.addWidget(self.config_button)

        self.image_label = QLabel('Select a directory to apply augmentations')
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        self.load_button = QPushButton('Load Directory')
        self.load_button.clicked.connect(self.load_directory)
        layout.addWidget(self.load_button)

        self.apply_button = QPushButton('Apply Augmentations')
        self.apply_button.clicked.connect(self.apply_augmentations)
        self.apply_button.setEnabled(False)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def load_config_file(self):
        file_dialog = QFileDialog()
        config_file_path, _ = file_dialog.getOpenFileName(self, 'Select Config File', '', 'YAML Files (*.yaml *.yml)')

        if config_file_path:
            self.config_file_path = config_file_path
            self.config_label.setText(f'Loaded config file: {os.path.basename(config_file_path)}')
            if hasattr(self, 'directory_path'):
                self.apply_button.setEnabled(True)

    def load_directory(self):
        file_dialog = QFileDialog()
        directory_path = file_dialog.getExistingDirectory(self, 'Select Directory')

        if directory_path:
            self.directory_path = directory_path
            self.image_label.setText(f'Loaded directory: {os.path.basename(directory_path)}')
            self.image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.jpg')]
            if hasattr(self, 'config_file_path'):
                self.apply_button.setEnabled(True)

    def apply_augmentations(self):
        output_directory = self.directory_path + '_aug'
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        with open(self.config_file_path, 'r') as f:
            yaml = ruamel.yaml.YAML()
            algorithms = yaml.load(f)

        self.original_augmented_pairs = []
        for image_path in self.image_paths:
            image = cv2.imread(image_path)
            augmented_images = []
            for alg in algorithms:
                augmented_image = image.copy()
                if isinstance(alg['augmentation'], list):
                    for i, a in enumerate(alg['augmentation']):
                        parameters = alg['parameters'][i] if 'parameters' in alg else None
                        augmented_image = apply(augmented_image, a, parameters)
                else:
                    parameters = alg['parameters'] if 'parameters' in alg else None
                    augmented_image = apply(augmented_image, alg['augmentation'], parameters)
                augmented_images.append(augmented_image)
                # Save each augmented image
                suffix = '_'.join(alg['augmentation']) if isinstance(alg['augmentation'], list) else alg['augmentation']
                output_path = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(image_path))[0]}_{suffix}.jpg")
                cv2.imwrite(output_path, augmented_image)
            self.original_augmented_pairs.append((image, augmented_images))

        self.current_index = 0
        self.show_comparison_window()

    def show_comparison_window(self):
        if not hasattr(self, 'comparison_window'):
            self.comparison_window = QtWidgets.QWidget()
            self.comparison_window.setWindowTitle('Image Comparison')
            self.comparison_window.setGeometry(150, 150, 1200, 800)

            self.fig = plt.figure()  # Removed figsize parameter
            self.canvas = FigureCanvas(self.fig)

            layout = QVBoxLayout()
            layout.addWidget(self.canvas)

            nav_layout = QHBoxLayout()
            self.prev_button = QPushButton('Previous')
            self.prev_button.clicked.connect(self.show_previous_image)
            self.next_button = QPushButton('Next')
            self.next_button.clicked.connect(self.show_next_image)

            nav_layout.addWidget(self.prev_button)
            nav_layout.addWidget(self.next_button)

            layout.addLayout(nav_layout)
            self.comparison_window.setLayout(layout)

        self.update_comparison_images()
        self.comparison_window.show()

    def update_comparison_images(self):
        original_image, augmented_images = self.original_augmented_pairs[self.current_index]

        self.fig.clear()

        num_images = len(augmented_images) + 1
        axes = self.fig.subplots(1, num_images, squeeze=False)[0] if num_images > 1 else [self.fig.subplots(1, 1)[0]]

        for ax in axes:
            ax.axis('off')

        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image", fontsize=10)

        for i, augmented_image in enumerate(augmented_images):
            axes[i + 1].imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(f"Augmented Image {i + 1}", fontsize=10)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_comparison_images()

    def show_next_image(self):
        if self.current_index < len(self.original_augmented_pairs) - 1:
            self.current_index += 1
            self.update_comparison_images()
