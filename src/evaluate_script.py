"""
evaluate_script.py
Tests the trained script classifier on the test set, and runs two shortcut
checks: padded vs full windows, and blanked top and bottom edges.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.datasets import get_script_dataloaders
from src.models.script_cnn import ScriptCNN
from src.utils import get_device, load_config, set_seed

EDGE_ROWS = 6              # rows blanked at the top and at the bottom for the edge check
MAX_MISTAKES_SHOWN = 40    # maximum number of mistakes drawn in the image grid


def load_script_model(config, device):
    """Rebuild the model from the checkpoint's own settings and load its weights."""
    path = config["script_classifier"]["checkpoint_path"]
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    languages = checkpoint["languages"]
    assert languages == list(config["data"]["languages"]), (
        f"Language order in the checkpoint {languages} does not match the config. "
        "Restore the original order in config.yaml or retrain the model."
    )

    settings = checkpoint["model_settings"]
    model = ScriptCNN(
        num_classes=len(languages),
        channels=settings["channels"],
        dropout=settings["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(validation accuracy {checkpoint['val_accuracy']:.2%})")
    return model, languages


def get_predictions(model, loader, device, blank_rows=0):
    """Predict a class for every window, optionally blanking the top and bottom rows first."""
    all_preds = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            if blank_rows > 0:
                images[:, :, :blank_rows, :] = 0
                images[:, :, -blank_rows:, :] = 0
            outputs = model(images.to(device))
            all_preds.append(outputs.argmax(dim=1).cpu())
    return torch.cat(all_preds).tolist()


def save_confusion_matrix_plot(matrix, path):
    """Draw the confusion matrix (a pandas table) as a colored grid and save it."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(matrix.values, cmap="Blues")

    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Predicted script")
    ax.set_ylabel("True script")
    ax.set_title("Script classifier: confusion matrix (test set)")

    threshold = matrix.values.max() / 2
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.values[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, f"{value:,}", ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_mistakes_plot(dataset, mistakes, path):
    """Draw the windows the model got wrong, titled with their true and predicted scripts."""
    shown = mistakes.head(MAX_MISTAKES_SHOWN)
    columns = 5
    rows = math.ceil(len(shown) / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(15, 1.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, (index, row) in zip(axes.flat, shown.iterrows()):
        image, _ = dataset[index]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"{row['language']} -> {row['predicted']}", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    config = load_config()
    set_seed(config["seed"])
    device = get_device()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    _, _, test_loader = get_script_dataloaders(config)
    model, languages = load_script_model(config, device)

    # One row per test window, in the same order the DataLoader serves them
    results = test_loader.dataset.rows.copy()
    predictions = get_predictions(model, test_loader, device)
    results["predicted"] = [languages[p] for p in predictions]
    results["correct"] = results["predicted"] == results["language"]

    # 1. Test accuracy
    n_mistakes = (~results["correct"]).sum()
    print(f"\nTest accuracy: {results['correct'].mean():.2%} "
          f"({n_mistakes} mistakes out of {len(results):,})")
    per_language = (results.groupby("language")["correct"].mean() * 100).round(2)
    print("\nAccuracy per language (%):")
    print(per_language.reindex(languages).to_string())

    # 2. Confusion matrix
    matrix = pd.crosstab(results["language"], results["predicted"])
    matrix = matrix.reindex(index=languages, columns=languages, fill_value=0)
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(matrix)
    save_confusion_matrix_plot(matrix, out_dir / "script_confusion_matrix.png")

    # 3. Shortcut check 1: padded vs full windows
    window_width = config["script_classifier"]["window_width"]
    results["window"] = (results["processed_width"] < window_width).map(
        {True: "padded", False: "full"}
    )
    padding = results.groupby(["language", "window"])["correct"].agg(
        windows="count", accuracy="mean"
    )
    padding["accuracy"] = (padding["accuracy"] * 100).round(2)
    print("\nShortcut check 1: padded vs full windows")
    print(padding)

    padded = results[results["window"] == "padded"]
    print("\nPredictions for padded windows (rows = true, columns = predicted):")
    print(pd.crosstab(padded["language"], padded["predicted"]))

    # 4. Shortcut check 2: blank the top and bottom edges
    blanked = get_predictions(model, test_loader, device, blank_rows=EDGE_ROWS)
    results["predicted_blanked"] = [languages[p] for p in blanked]
    results["correct_blanked"] = results["predicted_blanked"] == results["language"]

    edge = pd.DataFrame({
        "normal": results.groupby("language")["correct"].mean(),
        "edges_blanked": results.groupby("language")["correct_blanked"].mean(),
    }).reindex(languages)
    edge.loc["overall"] = [results["correct"].mean(), results["correct_blanked"].mean()]
    print(f"\nShortcut check 2: accuracy (%) with the top and bottom {EDGE_ROWS} rows blanked")
    print((edge * 100).round(2))

    # 5. The mistakes
    mistakes = results[~results["correct"]]
    print(f"\nMistakes ({len(mistakes)}):")
    print(mistakes[["language", "predicted", "processed_width", "label"]].to_string())
    if len(mistakes) > 0:
        save_mistakes_plot(test_loader.dataset, mistakes, out_dir / "script_mistakes.png")

    print(f"\nSaved: {out_dir / 'script_confusion_matrix.png'}")
    print(f"Saved: {out_dir / 'script_mistakes.png'}")


if __name__ == "__main__":
    main()