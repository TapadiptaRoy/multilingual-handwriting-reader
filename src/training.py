"""
training.py
Reusable training and evaluation functions for classification models.
"""

import time

import torch


def train_one_epoch(model, loader, loss_fn, optimizer, device, print_every=500):
    """Train the model on every batch once. Return average loss and accuracy."""
    model.train()
    total_loss, total_correct, total_images = 0.0, 0, 0
    start = time.time()

    for batch_number, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)              # 1. Predict
        loss = loss_fn(outputs, labels)      # 2. Measure how wrong
        optimizer.zero_grad()                # 3. Clear old gradients
        loss.backward()                      # 4. Calculate new gradients
        optimizer.step()                     # 5. Update the weights

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_images += batch_size

        if print_every and batch_number % print_every == 0:
            seconds = time.time() - start
            print(
                f"    batch {batch_number:>5,}/{len(loader):,} | "
                f"loss {total_loss / total_images:.4f} | "
                f"acc {total_correct / total_images:.2%} | {seconds:.0f}s"
            )

    return total_loss / total_images, total_correct / total_images


def evaluate(model, loader, loss_fn, device):
    """Check the model on a dataset without learning. Return average loss and accuracy."""
    model.eval()
    total_loss, total_correct, total_images = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_images += batch_size

    return total_loss / total_images, total_correct / total_images