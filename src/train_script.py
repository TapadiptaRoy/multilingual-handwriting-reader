"""
train_script.py
Trains the script classifier (Hindi / Bengali / Tamil), validating after every
epoch, logging to TensorBoard, and saving the best model.
"""

import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.datasets import get_script_dataloaders
from src.models.script_cnn import build_script_model
from src.training import evaluate, train_one_epoch
from src.utils import get_device, load_config, set_seed


def main():
    """Load everything, train for several epochs, and save the best model."""
    config = load_config()
    cls_cfg = config["script_classifier"]
    languages = list(config["data"]["languages"])

    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")
    print(f"Classes: {languages}")

    # Data, model, loss, optimizer, and learning rate schedule
    train_loader, val_loader, _ = get_script_dataloaders(config)
    model = build_script_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cls_cfg["learning_rate"])
    epochs = cls_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # TensorBoard: one folder per run, named with the date and time
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=Path(cls_cfg["log_dir"]) / run_name)

    checkpoint_path = Path(cls_cfg["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start = time.time()
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{epochs} (learning rate {learning_rate:.6f})")

        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        # Log this epoch's results for TensorBoard
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("accuracy/train", train_acc, epoch)
        writer.add_scalar("accuracy/val", val_acc, epoch)
        writer.add_scalar("learning_rate", learning_rate, epoch)

        minutes = (time.time() - start) / 60
        message = (
            f"  train loss {train_loss:.4f}, acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.2%} | {minutes:.1f} min"
        )

        # Save the best model, together with everything needed to use it later
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "languages": languages,
                    "model_settings": {
                        "channels": cls_cfg["channels"],
                        "dropout": cls_cfg["dropout"],
                    },
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )
            message += " | saved best model"

        print(message)

    writer.close()
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2%}")
    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()