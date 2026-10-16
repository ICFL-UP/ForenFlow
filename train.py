"""Shared training loop for all four architectures."""
import copy
import json
import os

import torch
from torch.utils.data import DataLoader

import config


def history_path(model_name, save_dir=config.MODEL_DIR):
    if save_dir == config.MODEL_DIR:
        return config.RESULTS_DIR / f"{model_name}_history.json"
    return save_dir / f"{model_name}_history.json"


def _run_epoch(model, loader, criterion, device, is_autoencoder, optimizer=None):
    running_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).squeeze(-1)
        if optimizer:
            optimizer.zero_grad()

        if is_autoencoder:
            loss = criterion(model(X_batch), X_batch)
        else:
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            # 0.5 here is only a progress indicator during training the operating threshold
            # is chosen afterwards with Youden J on the validation ROC curve
            correct += ((torch.sigmoid(logits) >= 0.5).float() == y_batch).sum().item()
            total += y_batch.size(0)

        if optimizer:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset), (correct / total if total else 0.0)


def train_model(model, train_loader: DataLoader, val_loader: DataLoader, epochs,
                lr=config.LEARNING_RATE, patience=config.EARLY_STOP_PATIENCE,
                model_name="model", is_autoencoder=False, save_dir=config.MODEL_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = model.loss_fn()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device,
                                           is_autoencoder, optimizer)
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, is_autoencoder)

        for key, value in (("train_loss", train_loss), ("val_loss", val_loss),
                           ("train_acc", train_acc), ("val_acc", val_acc)):
            history[key].append(value)

        status = f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        if not is_autoencoder:
            status += f" | train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        print(f"    {status}")

        if val_loss < best_val_loss:
            best_val_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(save_dir, exist_ok=True)
    path = save_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), path)
    with open(history_path(model_name, save_dir), "w") as f:
        json.dump(history, f, indent=2)
    print(f"    Model saved -> {path}\n")
    return history, model
