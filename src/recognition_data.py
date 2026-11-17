"""
recognition_data.py
Dataset and DataLoaders for the handwriting recognizer: serves whole word
images, pads them to the same width within each batch, and packs labels in
the format CTC loss expects.
"""

import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class WordDataset(Dataset):
    """Serves one full word image, its encoded label, and its text."""

    def __init__(self, manifest, language, split, charset, train, aug_cfg=None):
        is_selected = (manifest["language"] == language) & (manifest["split"] == split)
        self.rows = manifest[is_selected].reset_index(drop=True)
        self.charset = charset
        self.train = train
        self.aug_cfg = aug_cfg

        if train:
            shear = aug_cfg["shear"]
            self.affine = v2.RandomAffine(
                degrees=aug_cfg["rotation"],
                shear=(-shear, shear, 0, 0),   # slant left or right only
                fill=0,                        # fill new areas with background
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]

        with Image.open(row["processed_path"]) as img:
            if self.train:
                # Augmentation 1: stretch or squeeze the width slightly
                low, high = self.aug_cfg["width_scale"]
                new_width = max(1, round(img.width * random.uniform(low, high)))
                img = img.resize((new_width, img.height), Image.Resampling.BILINEAR)
            image = np.array(img, dtype=np.float32) / 255.0

        image = torch.from_numpy(1.0 - image).unsqueeze(0)   # invert; shape (1, 64, width)

        if self.train:
            # Augmentation 2: slight rotation and slant
            image = self.affine(image)

        target = torch.tensor(self.charset.encode(row["label"]), dtype=torch.long)
        return image, target, row["label"]


def collate_words(batch):
    """Combine words of different widths into one padded batch, with labels packed for CTC."""
    images, targets, texts = zip(*batch)

    widths = torch.tensor([image.shape[-1] for image in images], dtype=torch.long)
    height = images[0].shape[1]

    padded = torch.zeros(len(images), 1, height, int(widths.max()))
    for i, image in enumerate(images):
        padded[i, :, :, : image.shape[-1]] = image

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)

    return padded, targets, target_lengths, widths, list(texts)


def get_recognition_dataloaders(config, charset):
    """Create train, validation, and test DataLoaders for the recognizer's language."""
    rec_cfg = config["recognizer"]
    language = rec_cfg["language"]

    manifest = pd.read_csv(
        config["preprocess"]["processed_manifest_path"],
        encoding="utf-8",
        keep_default_na=False,
    )

    train_set = WordDataset(manifest, language, "train", charset, train=True, aug_cfg=rec_cfg["augment"])
    val_set = WordDataset(manifest, language, "val", charset, train=False)
    test_set = WordDataset(manifest, language, "test", charset, train=False)

    num_workers = rec_cfg["num_workers"]
    loader_settings = {
        "batch_size": rec_cfg["batch_size"],
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_words,
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_settings)
    val_loader = DataLoader(val_set, shuffle=False, **loader_settings)
    test_loader = DataLoader(test_set, shuffle=False, **loader_settings)

    return train_loader, val_loader, test_loader


def slices_needed(label):
    """Minimum CTC slices for a label: one per character, plus one blank between repeats."""
    repeats = sum(1 for a, b in zip(label, label[1:]) if a == b)
    return len(label) + repeats


if __name__ == "__main__":
    # This test only runs when the file is run directly, not when it is imported.
    import matplotlib.pyplot as plt

    from src.charset import Charset
    from src.utils import load_config

    config = load_config()
    rec_cfg = config["recognizer"]
    charset = Charset.load(rec_cfg["charset_path"])
    train_loader, val_loader, test_loader = get_recognition_dataloaders(config, charset)

    print(f"Language: {rec_cfg['language']}")
    print(f"Training words:   {len(train_loader.dataset):,}")
    print(f"Validation words: {len(val_loader.dataset):,}")
    print(f"Test words:       {len(test_loader.dataset):,}")

    # 1. Check one batch
    images, targets, target_lengths, widths, texts = next(iter(train_loader))
    print(f"\nBatch images shape:     {tuple(images.shape)}")
    print(f"All targets joined:     {tuple(targets.shape)}  (sum of target lengths = {target_lengths.sum().item()})")
    print(f"First 8 target lengths: {target_lengths[:8].tolist()}")
    print(f"First 8 real widths:    {widths[:8].tolist()}")

    # 2. Check the slice rule for every training word, in the worst case (narrowest stretch)
    reduction = rec_cfg["width_reduction"]
    narrowest = rec_cfg["augment"]["width_scale"][0]
    rows = train_loader.dataset.rows
    available = (rows["processed_width"] * narrowest).astype(int) // reduction
    needed = rows["label"].map(slices_needed)
    too_short = rows[available < needed]
    print(f"\nTraining words with too few slices (worst case): {len(too_short)} of {len(rows):,}")
    if len(too_short) > 0:
        print(too_short[["label", "processed_width"]].head(10).to_string())

    # 3. Show three training words: original, then three random augmentations of each
    manifest = pd.read_csv(config["preprocess"]["processed_manifest_path"], encoding="utf-8", keep_default_na=False)
    plain_train = WordDataset(manifest, rec_cfg["language"], "train", charset, train=False)
    augmented_train = train_loader.dataset

    fig, axes = plt.subplots(3, 4, figsize=(14, 5))
    for row_number in range(3):
        original, _, _ = plain_train[row_number]
        axes[row_number, 0].imshow(original.squeeze(), cmap="gray")
        axes[row_number, 0].set_title("original")
        for column in range(1, 4):
            augmented, _, _ = augmented_train[row_number]
            axes[row_number, column].imshow(augmented.squeeze(), cmap="gray")
            axes[row_number, column].set_title(f"augmented {column}")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()