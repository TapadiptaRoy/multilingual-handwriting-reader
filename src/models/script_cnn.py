"""
script_cnn.py
CNN that classifies a handwritten word window by script (language).
"""

import torch
from torch import nn


def conv_block(in_channels, out_channels):
    """Conv -> BatchNorm -> ReLU -> MaxPool. Halves the height and width."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


class ScriptCNN(nn.Module):
    """Four Conv blocks, global average pooling, and a Linear layer."""

    def __init__(self, num_classes, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()

        # Part 1: feature extractor, built from one Conv block per channel size
        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        # Part 2: classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # global average pooling: (C, H, W) -> (C, 1, 1)
            nn.Flatten(),                            # (C, 1, 1) -> (C)
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),    # (256) -> (3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_script_model(config):
    """Create a ScriptCNN using the settings from the config file."""
    cls_cfg = config["script_classifier"]
    return ScriptCNN(
        num_classes=len(config["data"]["languages"]),
        channels=cls_cfg["channels"],
        dropout=cls_cfg["dropout"],
    )


def count_parameters(model):
    """Count how many numbers (weights) the model can learn."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # This test only runs when the file is run directly, not when it is imported.
    from src.utils import load_config

    config = load_config()
    model = build_script_model(config)

    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Pass a fake batch through the model to check the shapes
    window_width = config["script_classifier"]["window_width"]
    fake_windows = torch.randn(8, 1, 64, window_width)
    features = model.features(fake_windows)
    output = model(fake_windows)

    print(f"Input shape:          {tuple(fake_windows.shape)}")
    print(f"After the Conv blocks: {tuple(features.shape)}")
    print(f"Output shape:         {tuple(output.shape)}")