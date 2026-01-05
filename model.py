import torch
import torch.nn as nn
import torchvision.models as models

class MedicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ResNet18 structure
        # Note: We don't need pretrained weights=True here since we are loading our own trained weights
        self.model = models.resnet18(pretrained=False)

        # Reconstruct the architecture exactly as it was during training
        self.model.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.model(x)