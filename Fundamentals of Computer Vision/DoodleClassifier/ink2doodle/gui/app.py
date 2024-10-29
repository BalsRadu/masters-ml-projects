import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                             QVBoxLayout, QWidget)
from torchvision import transforms


class ScratchPad(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(560, 560)
        self.image = QImage(560, 560, QImage.Format.Format_RGB32)
        self.image.fill(QColor('#000000'))
        self.drawing = False
        self.erasing = False
        self.last_point = QPoint()
        self.pen_width = 28  # Increased pen width
        self.alpha = 255
        self.fade_step = 15  # Controls how quickly the line fades
        self.min_alpha = 80  # Minimum opacity for the gradient

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()
            self.alpha = 255  # Reset to full intensity
        elif event.button() == Qt.MouseButton.RightButton:
            self.erasing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if self.drawing or self.erasing:
            painter = QPainter(self.image)
            current_point = event.position().toPoint()
            
            if self.drawing:
                # Calculate distance for more natural fading
                distance = ((current_point.x() - self.last_point.x()) ** 2 + 
                          (current_point.y() - self.last_point.y()) ** 2) ** 0.5
                
                # Adjust alpha based on distance
                self.alpha = max(self.min_alpha, 
                               self.alpha - (self.fade_step * (distance / 10)))
                
                painter.setPen(QPen(QColor(255, 255, 255, int(self.alpha)), 
                                  self.pen_width, 
                                  Qt.PenStyle.SolidLine,
                                  Qt.PenCapStyle.RoundCap,
                                  Qt.PenJoinStyle.RoundJoin))
            elif self.erasing:
                painter.setPen(QPen(QColor(0, 0, 0),
                                  self.pen_width,
                                  Qt.PenStyle.SolidLine,
                                  Qt.PenCapStyle.RoundCap,
                                  Qt.PenJoinStyle.RoundJoin))
                
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.erasing = False

    def clear(self):
        self.image.fill(QColor('#000000'))
        self.update()

    def get_image(self):
        img = self.image.scaled(28, 28, aspectRatioMode=Qt.AspectRatioMode.IgnoreAspectRatio)
        img = img.convertToFormat(QImage.Format.Format_Grayscale8)
        ptr = img.bits()
        ptr.setsize(img.width() * img.height())
        img_array = np.array(ptr, dtype=np.uint8).reshape((img.height(), img.width()))
        pil_image = Image.fromarray(img_array)
        return pil_image

class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.setup_ui()
        self.setup_timer()

    def setup_ui(self):
        # Window setup
        self.setWindowTitle("Ink2Doodle")
        self.setGeometry(100, 100, 900, 650)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #2c2c2c;
                color: #AAAAAA;
            }
        """)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left side (drawing area and instructions)
        left_layout = QVBoxLayout()
        
        # Drawing area (ScratchPad)
        self.scratchpad = ScratchPad(self)
        self.scratchpad.setStyleSheet("""
            ScratchPad {
                background-color: #000000;
                border: 4px solid #444444;
                border-radius: 4px;
            }
        """)
        
        # Instructions label
        self.instructions_label = QLabel("Left mouse: Draw | Right mouse: Erase | C: Clear")
        self.instructions_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 14px;
                padding: 10px 0;
            }
        """)
        self.instructions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_layout.addWidget(self.scratchpad)
        left_layout.addWidget(self.instructions_label)

        # Right side (predictions)
        right_widget = QWidget()
        right_widget.setFixedWidth(300)  # Fixed width for prediction area
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(20, 0, 0, 0)  # Add left padding
        right_layout.setSpacing(0)  # Remove spacing between predictions
        
        # Container to hold predictions and stretch them vertically
        predictions_container = QWidget()
        predictions_layout = QVBoxLayout(predictions_container)
        predictions_layout.setSpacing(0)
        predictions_layout.setContentsMargins(0, 0, 0, 0)

        # Create labels for predictions
        self.prediction_labels = []
        for i in range(25):
            prediction_widget = QWidget()
            prediction_layout = QHBoxLayout(prediction_widget)
            prediction_layout.setContentsMargins(0, 0, 0, 0)
            prediction_layout.setSpacing(10)

            # Class name label
            class_label = QLabel()
            class_label.setFont(QFont("Courier New", 14))  # Increased font size
            class_label.setStyleSheet(f"""
                color: {'#FFFFFF' if i == 0 else '#AAAAAA'};
                font-weight: {'bold' if i == 0 else 'normal'};
            """)
            
            # Percentage label
            percentage_label = QLabel()
            percentage_label.setFont(QFont("Courier New", 14))  # Increased font size
            percentage_label.setStyleSheet(f"""
                color: {'#FFFFFF' if i == 0 else '#AAAAAA'};
                font-weight: {'bold' if i == 0 else 'normal'};
            """)
            percentage_label.setAlignment(Qt.AlignmentFlag.AlignRight)

            prediction_layout.addWidget(class_label, 1)  # 1 is the stretch factor
            prediction_layout.addWidget(percentage_label)
            
            self.prediction_labels.append((class_label, percentage_label))
            predictions_layout.addWidget(prediction_widget)

        # Add stretch to evenly distribute space
        predictions_layout.addStretch()
        
        # Add the predictions container to the right layout
        right_layout.addWidget(predictions_container, 1)  # 1 is the stretch factor

        # Add layouts to main layout
        main_layout.addLayout(left_layout)
        main_layout.addWidget(right_widget)

        self.setCentralWidget(main_widget)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_predictions)
        self.timer.start(100)  # Update every 100ms

    def update_predictions(self):
        img = self.scratchpad.get_image()
        img_tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top_k_probs, top_k_indices = torch.topk(probabilities, 25)
            top_k_probs = top_k_probs.cpu().numpy()
            top_k_indices = top_k_indices.cpu().numpy()

        # Load class names
        with open("categories.txt", "r") as f:
            classes = [line.strip().replace('_', ' ').title() for line in f.readlines()]

        # Update prediction labels
        for i in range(25):
            class_name = classes[top_k_indices[i]]
            confidence = top_k_probs[i] * 100
            
            class_label, percentage_label = self.prediction_labels[i]
            class_label.setText(f"{class_name}")
            percentage_label.setText(f"{confidence:>6.2f}%")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_C:
            self.scratchpad.clear()
