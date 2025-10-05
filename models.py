import torch.nn as nn
import torchvision.models as tv_models
from efficientnet_pytorch import EfficientNet  # if you installed efficientnet_pytorch

class TestModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 128 * 128, num_classes)  # adjust based on your spectrogram size

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def create_model(backbone: str, in_channels: int = 3, num_classes: int = 6, pretrained: bool = False):
    """
    Factory function to build models dynamically by backbone name.
    """
    if backbone == "resnet50":
        model = tv_models.resnet50(pretrained=pretrained)
        # change first conv layer to match in_channels
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # change fc for num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif backbone == "efficientnet_b0":
        model = tv_models.efficientnet_b0(pretrained=pretrained)
        if in_channels != 3:
            model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    elif backbone == "test":
        return TestModel(in_channels=in_channels, num_classes=num_classes)

    else:
        raise ValueError(f"Backbone {backbone} not supported")
