"""Model evaluation: Youden J thresholds, test metrics, plots and global SHAP values."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_curve)

from config import PLOT_DIR, SHAP_BACKGROUND_SAMPLES, SHAP_MAX_DISPLAY, FEATURE_COLUMNS


def predict_scores(model, loader, is_autoencoder=False):
    """Per-sequence score: sigmoid probability for a classifier, reconstruction MSE for the
    autoencoder. Higher always means 'more like an attack'."""
    device = next(model.parameters()).device
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            if is_autoencoder:
                batch = ((model(X) - X) ** 2).mean(dim=(1, 2))
            else:
                batch = torch.sigmoid(model(X))
            scores.append(batch.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(scores).ravel(), np.concatenate(labels).ravel()


def youden_threshold(y_true, scores):
    """Operating point that maximises J = TPR - FPR on the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best = np.argmax(tpr - fpr)
    return float(thresholds[best]), float(tpr[best] - fpr[best]), fpr, tpr, best


def _save(fig, filename, plot_dir=PLOT_DIR):
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_roc(fpr, tpr, op_fpr, op_tpr, title, point_label, filename, plot_dir=PLOT_DIR):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
    ax.plot(op_fpr, op_tpr, "ro", label=point_label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, filename, plot_dir)


def plot_training_curves(history, model_name, plot_dir=PLOT_DIR):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if any(v > 0 for v in history["train_acc"]):
        axes[1].plot(epochs, history["train_acc"], label="Train")
        axes[1].plot(epochs, history["val_acc"], label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{model_name} - Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "N/A (autoencoder)", ha="center", va="center", fontsize=12)
        axes[1].set_title(f"{model_name} - Accuracy")

    plt.tight_layout()
    return _save(fig, f"{model_name}_training_curves.png", plot_dir)


def plot_confusion_matrix(y_true, y_pred, model_name, plot_dir=PLOT_DIR):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    return _save(fig, f"{model_name}_confusion_matrix.png", plot_dir)


def plot_classification_report(y_true, y_pred, model_name, plot_dir=PLOT_DIR):
    report_text = classification_report(y_true, y_pred, target_names=["Normal", "Attack"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.05, 0.95, report_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace")
    ax.set_title(f"{model_name} - Classification Report", fontsize=12)
    plt.tight_layout()
    return _save(fig, f"{model_name}_classification_report.png", plot_dir), report_text


def plot_error_distribution(errors, y_true, threshold, model_name, plot_dir=PLOT_DIR):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors[y_true == 0], bins=50, alpha=0.6, label="Normal", density=True)
    ax.hist(errors[y_true == 1], bins=50, alpha=0.6, label="Attack", density=True)
    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} - Reconstruction Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, f"{model_name}_error_dist.png", plot_dir)


def select_youden_threshold(model, val_loader, model_name, is_autoencoder=False, plot_dir=PLOT_DIR):
    """Threshold from the VALIDATION set only"""
    print(f"  Selecting Youden J threshold for {model_name} on validation set ...")
    scores, y_true = predict_scores(model, val_loader, is_autoencoder)
    threshold, j_value, fpr, tpr, best = youden_threshold(y_true, scores)
    _plot_roc(fpr, tpr, fpr[best], tpr[best], f"{model_name} - Youden J (Validation)",
              f"Youden T = {threshold:.4f}", f"{model_name}_youden_roc.png", plot_dir)
    print(f"    Youden J threshold = {threshold:.4f} (J = {j_value:.4f}, val AUC = {auc(fpr, tpr):.4f})")
    return threshold, j_value


def evaluate_model(model, test_loader, history, model_name, threshold, is_autoencoder=False,
                   plot_dir=PLOT_DIR):
    """Test-set metrics at the validation-derived threshold, plus every plot Chapter 5 uses.
    plot_dir defaults to the shared plot directory"""
    print(f"  Evaluating {model_name} ...")
    scores, y_true = predict_scores(model, test_loader, is_autoencoder)
    y_pred = (scores >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, scores)
    plots = {
        "roc": _plot_roc(fpr, tpr,
                         np.mean(scores[y_true == 0] >= threshold),
                         np.mean(scores[y_true == 1] >= threshold),
                         f"{model_name} - ROC Curve", f"Threshold = {threshold:.4f}",
                         f"{model_name}_roc_curve.png", plot_dir),
        "confusion_matrix": plot_confusion_matrix(y_true, y_pred, model_name, plot_dir),
        "training_curves": plot_training_curves(history, model_name, plot_dir),
    }
    plots["classification_report"], report_text = plot_classification_report(
        y_true, y_pred, model_name, plot_dir)
    if is_autoencoder:
        plots["error_distribution"] = plot_error_distribution(
            scores, y_true, threshold, model_name, plot_dir)

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc(fpr, tpr),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "report_text": report_text,
    }
    print(f"    Accuracy={metrics['accuracy']:.4f}  Precision={metrics['precision']:.4f}  "
          f"Recall={metrics['recall']:.4f}  F1={metrics['f1_score']:.4f}  AUC={metrics['auc']:.4f}")
    return metrics, plots


def compute_shap_values(model, train_loader, model_name="LSTM_Improved", plot_dir=PLOT_DIR):
    """Global feature importance with GradientExplainer over 100 background sequences."""
    import shap

    print(f"  Computing SHAP values for {model_name} ...")
    device = next(model.parameters()).device
    model.eval()

    # GradientExplainer expects a 2-D output the model squeezes to 1-D
    class _Unsqueezed(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x).unsqueeze(-1)

    def take(n):
        batches, count = [], 0
        for X, _ in train_loader:
            batches.append(X)
            count += X.shape[0]
            if count >= n:
                break
        return torch.cat(batches, dim=0)[:n].to(device)

    explainer = shap.GradientExplainer(_Unsqueezed(model).to(device).eval(),
                                       take(SHAP_BACKGROUND_SAMPLES))
    shap_values = explainer.shap_values(take(500))
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_array = np.asarray(shap_values)
    # newer shap releases keep the output dimension: (samples, steps, features, 1)
    if shap_array.ndim == 4:
        shap_array = shap_array[..., 0]
    # mean |SHAP| per feature across samples and sequence steps
    mean_abs = np.mean(np.abs(shap_array), axis=(0, 1))
    shap_dict = dict(sorted(
        ((FEATURE_COLUMNS[i], float(mean_abs[i])) for i in range(len(FEATURE_COLUMNS))),
        key=lambda kv: kv[1], reverse=True))

    os.makedirs(plot_dir, exist_ok=True)
    shap_path = os.path.join(plot_dir, "shap_values.json")
    with open(shap_path, "w") as f:
        json.dump(shap_dict, f, indent=2)
    print(f"    SHAP values saved -> {shap_path}")

    top_n = min(SHAP_MAX_DISPLAY, len(FEATURE_COLUMNS))
    features = list(shap_dict)[:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(top_n), [shap_dict[f] for f in features][::-1], color="#2c3e50")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(features[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{model_name} - SHAP Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    print(f"    SHAP plot saved -> {_save(fig, f'{model_name}_shap_importance.png', plot_dir)}")
    return shap_dict
