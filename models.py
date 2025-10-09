import torch
import torch.nn as nn
import torchvision.models as tvm


class TestModel(nn.Module):
    """Tiny conv + GAP baseline; returns [B, out_channels] logits."""

    def __init__(self, in_channels, out_channels):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):  # x: [B, in_ch, H, W]
        x = self.conv(x)  # [B, out_ch, H, W]
        x = self.pool(x)  # [B, out_ch, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B, out_ch]
        return x


def _replace_first_conv_for_in_channels(model, in_channels: int):
    """Utility: adjust first conv to accept custom in_channels (default 3)."""
    if in_channels == 3:
        return model
    # Handle common backbones (resnet/vgg features conv1)
    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        old = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
    elif hasattr(model, "features") and isinstance(model.features[0], nn.Conv2d):
        old = model.features[0]
        model.features[0] = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
    return model


def create_model(
    backbone: str, in_channels: int, num_classes: int, pretrained: bool = True
) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        m = _replace_first_conv_for_in_channels(m, in_channels)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if backbone == "resnet34":
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
        m = _replace_first_conv_for_in_channels(m, in_channels)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if backbone == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        m = _replace_first_conv_for_in_channels(m, in_channels)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if backbone == "vgg16":
        m = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT if pretrained else None)
        m = _replace_first_conv_for_in_channels(m, in_channels)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
        return m
    if backbone == "vgg11":
        m = tvm.vgg11(weights=tvm.VGG11_Weights.DEFAULT if pretrained else None)
        m = _replace_first_conv_for_in_channels(m, in_channels)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
        return m

    # Fallback tiny model
    return TestModel(in_channels, num_classes)
