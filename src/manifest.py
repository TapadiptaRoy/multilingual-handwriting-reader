"""
manifest.py
Reads each language's handwriting dataset in its own format and combines
everything into one table (the manifest), saved as a CSV file.
"""

import os
import unicodedata
from pathlib import Path
import pandas as pd
from src.utils import load_config

def normalize(text):
    """Convert text to the standard Unicode form (NFC)."""
    return unicodedata.normalize("NFC", text)

def read_iiit_hw_dev(folder, split):
    """Read one split of the IIIT-HW-Dev format (Hindi): 'path word' per line."""
    folder = Path(folder)
    rows = []

    with open(folder / f"{split}.txt", "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.split(" ", 1)
            rows.append({"image_path": (folder / path).as_posix(), "label": label})

    return rows

def read_iiit_indic(folder, split):
    """Read one split of the IIIT-INDIC-HW-WORDS format: 'path, vocab_id' per line."""
    folder = Path(folder)

    # Load the vocabulary: line number (starting at 0) -> word
    with open(folder / "vocab.txt", "r", encoding="utf-8-sig") as f:
        vocab = [line.strip() for line in f]

    rows = []
    with open(folder / f"{split}.txt", "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, vocab_id = line.split(", ")
            rows.append({"image_path": (folder / path).as_posix(), "label": vocab[int(vocab_id)]})

    return rows


# Which reading function to use for each dataset format
READERS = {
    "iiit_hw_dev": read_iiit_hw_dev,
    "iiit_indic": read_iiit_indic,
}


def build_manifest(config):
    """Read every language and split, check the data, and save one combined table."""
    data_cfg = config["data"]
    all_rows = []

    # 1. Read every language and split with the right reader
    print("Reading label files...")
    for language, lang_cfg in data_cfg["languages"].items():
        reader = READERS[lang_cfg["format"]]
        for split in data_cfg["splits"]:
            rows = reader(lang_cfg["folder"], split)
            for row in rows:
                row["language"] = language
                row["split"] = split
            all_rows.extend(rows)
            print(f"  {language:<8} {split:<5} {len(rows):>7,} rows")

    df = pd.DataFrame(all_rows, columns=["language", "split", "image_path", "label"])

    # 2. Normalize every label to NFC and count how many changed
    original = df["label"].copy()
    df["label"] = df["label"].map(normalize)
    changed = (df["label"] != original).sum()
    print(f"\nLabels changed by NFC normalization: {changed:,}")

    # 3. Check for empty labels
    empty = (df["label"] == "").sum()
    print(f"Empty labels: {empty:,}")

    # 4. Check that every image exists (this can take a minute)
    print("Checking that every image exists...")
    exists = df["image_path"].map(os.path.exists)
    missing = df[~exists]
    print(f"Missing images: {len(missing):,}")
    if len(missing) > 0:
        print("First few missing paths:")
        print(missing["image_path"].head().to_string(index=False))
        df = df[exists]
        print(f"Removed {len(missing):,} rows with missing images.")

    # 5. Save the manifest
    out_path = Path(data_cfg["manifest_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    # 6. Print a summary table
    print("\nImages per language and split:")
    summary = pd.crosstab(df["language"], df["split"])[data_cfg["splits"]]
    print(summary)
    print(f"\nTotal: {len(df):,} images")
    print(f"Manifest saved to: {out_path}")

    return df


if __name__ == "__main__":
    config = load_config()
    build_manifest(config)