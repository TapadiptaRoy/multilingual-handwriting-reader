"""
datasets.py
PyTorch Dataset for the script classifier: serves fixed-size windows of
handwritten word images, labeled by language (script).
"""

import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ScriptDataset(Dataset):
    """Serves 64 x window_width windows of word images, labeled by language."""

    def __init__(self, manifest, split, languages, window_width, train):
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.language_to_index = {language: i for i, language in enumerate(languages)}
        self.window_width = window_width
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]

        with Image.open(row["processed_path"]) as img:
            image = np.array(img, dtype=np.float32) / 255.0   # (64, width), 0 = black, 1 = white

        image = 1.0 - image                    # invert: background 0, ink bright
        image = self.take_window(image)        # (64, window_width)

        tensor = torch.from_numpy(image).unsqueeze(0)   # (1, 64, window_width)
        label = self.language_to_index[row["language"]]
        return tensor, label

    def take_window(self, image):
        """Cut a fixed-width window (random in training, centered in testing), padding if needed."""
        height, width = image.shape
        w = self.window_width

        # Wide enough: cut a window out of the word
        if width >= w:
            start = random.randint(0, width - w) if self.train else (width - w) // 2
            return image[:, start:start + w]

        # Too narrow: place the word inside an empty (zero) window
        window = np.zeros((height, w), dtype=np.float32)
        offset = random.randint(0, w - width) if self.train else (w - width) // 2
        window[:, offset:offset + width] = image
        return window


def get_script_dataloaders(config):
    """Create train, validation, and test DataLoaders for the script classifier."""
    cls_cfg = config["script_classifier"]
    languages = list(config["data"]["languages"])

    manifest = pd.read_csv(
        config["preprocess"]["processed_manifest_path"],
        encoding="utf-8",
        keep_default_na=False,
    )

    datasets = {
        "train": ScriptDataset(manifest, "train", languages, cls_cfg["window_width"], train=True),
        "val": ScriptDataset(manifest, "val", languages, cls_cfg["window_width"], train=False),
        "test": ScriptDataset(manifest, "test", languages, cls_cfg["window_width"], train=False),
    }

    num_workers = cls_cfg["num_workers"]
    loader_settings = {
        "batch_size": cls_cfg["batch_size"],
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(datasets["train"], shuffle=True, **loader_settings)
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_settings)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_settings)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # This test only runs when the file is run directly, not when it is imported.
    import matplotlib.pyplot as plt

    from src.utils import load_config

    config = load_config()
    languages = list(config["data"]["languages"])
    train_loader, val_loader, test_loader = get_script_dataloaders(config)

    print(f"Training windows:   {len(train_loader.dataset):,}")
    print(f"Validation windows: {len(val_loader.dataset):,}")
    print(f"Test windows:       {len(test_loader.dataset):,}")

    images, labels = next(iter(train_loader))
    print(f"\nBatch of images shape: {tuple(images.shape)}")
    print(f"Batch of labels shape: {tuple(labels.shape)}")
    print(f"Pixel values range:    {images.min().item():.2f} to {images.max().item():.2f}")

    fig, axes = plt.subplots(4, 2, figsize=(8, 7))
    for ax, image, label in zip(axes.flat, images, labels):
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(languages[label.item()])
        ax.axis("off")
    plt.tight_layout()
    plt.show()