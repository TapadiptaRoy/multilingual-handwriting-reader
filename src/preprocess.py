"""
preprocess.py
Converts every image in the manifest to grayscale, resizes it to a fixed height
(keeping the aspect ratio), saves it as PNG, and writes a new manifest that
points to the processed images.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from src.utils import load_config


def processed_path_for(image_path, output_folder):
    """Example: data/bengali/train/0.jpg -> data/processed/bengali/train/0.png"""
    relative = Path(image_path).relative_to("data")
    return (Path(output_folder) / relative).with_suffix(".png")


def process_one(task):
    """Process one image. Returns (new width, None) on success, or (None, error message)."""
    source, destination, height = task
    destination = Path(destination)

    try:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as img:
                gray = img.convert("L")
                new_width = max(1, round(gray.width * height / gray.height))
                resized = gray.resize((new_width, height), Image.Resampling.LANCZOS)
                resized.save(destination)

        with Image.open(destination) as img:
            return img.width, None

    except Exception as error:
        return None, f"{source}: {error}"


def main():
    config = load_config()
    pre_cfg = config["preprocess"]

    df = pd.read_csv(config["data"]["manifest_path"], encoding="utf-8", keep_default_na=False)

    # Work out where each processed image will be saved
    df["processed_path"] = [
        processed_path_for(path, pre_cfg["output_folder"]).as_posix()
        for path in df["image_path"]
    ]
    tasks = [
        (source, destination, pre_cfg["height"])
        for source, destination in zip(df["image_path"], df["processed_path"])
    ]

    # Process all images using several CPU cores
    print(f"Processing {len(tasks):,} images with {pre_cfg['num_workers']} workers...")
    start = time.time()
    widths, errors = [], []

    with ProcessPoolExecutor(max_workers=pre_cfg["num_workers"]) as executor:
        results = executor.map(process_one, tasks, chunksize=256)
        for count, (width, error) in enumerate(results, start=1):
            widths.append(width)
            if error:
                errors.append(error)
            if count % 10000 == 0 or count == len(tasks):
                minutes = (time.time() - start) / 60
                print(f"  {count:>7,} / {len(tasks):,} done  ({minutes:.1f} min)")

    # Record new widths and remove any images that failed
    df["processed_width"] = widths
    if errors:
        print(f"\n{len(errors):,} images failed. First few:")
        for error in errors[:5]:
            print("  ", error)
        df = df[df["processed_width"].notna()].copy()
    else:
        print("\nNo errors.")
    df["processed_width"] = df["processed_width"].astype(int)

    # Save the new manifest
    out_path = Path(pre_cfg["processed_manifest_path"])
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("\nProcessed widths (pixels) per language:")
    print(df.groupby("language")["processed_width"].describe().round(1))
    print(f"\nProcessed manifest saved to: {out_path}")


if __name__ == "__main__":
    main()