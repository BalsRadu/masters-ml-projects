from pathlib import Path

import torch
import torch.nn as nn

NUMBER_OF_CLASSES = 345

class Ink2Doodle:
    def __init__(self, model_path):
        # Resolves the model path
        model_path = Path(model_path).resolve()
        # Initialize the base model class
        self.model = Ink2DoodleNet(num_classes=NUMBER_OF_CLASSES)
        # Load checkpoint into memory
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        # Load state_dict into the model
        self.model.load_state_dict(checkpoint, strict=True)
        # Set the model to evaluation mode
        self.model.eval()

    def __call__(self, x):
        return self.model(x)

class Ink2DoodleNet(nn.Module):
    def __init__(self, num_classes=345):
        super(Ink2DoodleNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),

            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*3*3, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
