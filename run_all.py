#!/usr/bin/env python3
"""End-to-end ML pipeline: data, four models, Youden J thresholds, SHAP, comparison report."""
import json
import os
import sys

import torch

import config
from dataset import create_dataloaders, create_normal_only_loaders
from download_data import download_datasets
from evaluate import compute_shap_values, evaluate_model, select_youden_threshold
from generate_report import generate_report
from models.autoencoder import LSTMAutoencoder
from models.lstm_baseline import LSTMBaseline
from models.lstm_improved import LSTMImproved
from models.transformer import TransformerClassifier
from preprocess import preprocess_all
from train import train_model, history_path


def load_or_train(model, train_loader, val_loader, model_name, is_autoencoder=False):
    saved = config.MODEL_DIR / f"{model_name}.pt"
    if not saved.exists():
        return train_model(model, train_loader, val_loader, epochs=config.NUM_EPOCHS,
                           model_name=model_name, is_autoencoder=is_autoencoder)

    print(f"  Loading saved {model_name} from {saved} ...")
    model.load_state_dict(torch.load(saved, map_location="cpu", weights_only=True))
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if history_path(model_name).exists():
        with open(history_path(model_name)) as f:
            history = json.load(f)
    return history, model


def main():
    print("=" * 60)
    print("  F-RIaaS ML Experiment Pipeline")
    print("=" * 60)

    print("\n[Step 1] Downloading datasets from Kaggle ...")
    try:
        download_datasets()
    except Exception as exc:
        print(f"  Download failed: {exc}")
        if not any((config.RAW_DIR / f).exists()
                   for f in ["dataset1_intrusion.csv", "dataset2_luflow.csv"]):
            sys.exit("  No datasets found. Please set up Kaggle credentials and retry.")
        print("  Using existing datasets.")

    testbed_path = config.RAW_DIR / "dataset3_testbed.csv"
    if not testbed_path.exists():
        sys.exit(f"  No testbed capture at {testbed_path}. Run capture_dataset3.py on the host VM first.")
    print(f"\n[Step 2] Using existing testbed data: {testbed_path}")

    print("\n[Step 3] Preprocessing datasets ...")
    preprocess_all()

    print("\n[Step 4] Creating sequence datasets ...")
    train_loader, val_loader, test_loader, scaler = create_dataloaders()
    ae_train, ae_val = create_normal_only_loaders(scaler)

    # (name, model, training loaders, is_autoencoder) the autoencoder learns normal traffic only
    candidates = [
        ("Autoencoder", LSTMAutoencoder(), (ae_train, ae_val), True),
        ("Transformer", TransformerClassifier(), (train_loader, val_loader), False),
        ("LSTM_Baseline", LSTMBaseline(), (train_loader, val_loader), False),
        ("LSTM_Improved", LSTMImproved(), (train_loader, val_loader), False),
    ]

    all_metrics, all_plots, thresholds = [], [], {}
    for step, (name, model, (fit_train, fit_val), is_autoencoder) in enumerate(candidates, 5):
        print(f"\n[Step {step}] Training {name} ...")
        history, model = load_or_train(model, fit_train, fit_val, name, is_autoencoder)
        threshold, _ = select_youden_threshold(model, val_loader, name, is_autoencoder)
        metrics, plots = evaluate_model(model, test_loader, history, name, threshold, is_autoencoder)
        all_metrics.append(metrics)
        all_plots.append(plots)
        thresholds[name] = float(threshold)
        if name == "LSTM_Improved":
            print("\n[Step 8b] Computing SHAP feature attributions ...")
            compute_shap_values(model, train_loader, name)

    print("\n[Step 8c] Saving Youden J thresholds ...")
    os.makedirs(config.YOUDEN_RESULTS_FILE.parent, exist_ok=True)
    with open(config.YOUDEN_RESULTS_FILE, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"    Thresholds saved -> {config.YOUDEN_RESULTS_FILE}")

    print("\n[Step 9] Generating PDF report ...")
    report_path = generate_report(all_metrics, all_plots)

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\n  Plots     -> {config.PLOT_DIR}/")
    print(f"  Models    -> {config.MODEL_DIR}/")
    print(f"  Report    -> {report_path}\n")

    line = "  +---------------------+----------+----------+----------+----------+----------+"
    print(line)
    print("  | Model               | Accuracy | Precision| Recall   | F1-Score | AUC      |")
    print(line)
    for m in all_metrics:
        print(f"  | {m['model']:<19} | {m['accuracy']:.4f}   | {m['precision']:.4f}   |"
              f" {m['recall']:.4f}   | {m['f1_score']:.4f}   | {m['auc']:.4f}   |")
    print(line)
    best = max(all_metrics, key=lambda m: m["f1_score"])
    print(f"\n  Best model: {best['model']} (F1={best['f1_score']:.4f})")


if __name__ == "__main__":
    main()
