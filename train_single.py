#!/usr/bin/env python3
"""Train and evaluate one classifier inside its own run directory.

Usage:  python3 train_single.py --model LSTM_Improved --run-dir outputs/results/run13
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import config
from dataset import create_dataloaders
from evaluate import compute_shap_values, evaluate_model, select_youden_threshold
from models.lstm_baseline import LSTMBaseline
from models.lstm_improved import LSTMImproved
from models.transformer import TransformerClassifier
from train import train_model

CLASSIFIERS = {
    "LSTM_Improved": LSTMImproved,
    "LSTM_Baseline": LSTMBaseline,
    "Transformer": TransformerClassifier,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CLASSIFIERS), default="LSTM_Improved")
    parser.add_argument("--run-dir", default="outputs/results/run13")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = config.BASE_DIR / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print(f"  Fresh training run: {args.model} -> {run_dir}")
    print("=" * 60)

    print("\n[1] Creating sequence datasets ...")
    train_loader, val_loader, test_loader, _ = create_dataloaders(scaler_dir=run_dir)

    print(f"\n[2] Training {args.model} ...")
    history, model = train_model(CLASSIFIERS[args.model](), train_loader, val_loader,
                                 epochs=config.NUM_EPOCHS, model_name=args.model, save_dir=run_dir)

    print("\n[3] Selecting the operating threshold ...")
    threshold, j_value = select_youden_threshold(model, val_loader, args.model, plot_dir=run_dir)

    print("\n[4] Evaluating on the held-out test set ...")
    metrics, plots = evaluate_model(model, test_loader, history, args.model, threshold, plot_dir=run_dir)
    metrics["youden_j"] = j_value
    metrics["epochs_trained"] = len(history["train_loss"])
    metrics["seed"] = args.seed
    metrics["plots"] = plots

    if args.model == "LSTM_Improved":
        print("\n[5] Computing SHAP feature attributions ...")
        metrics["shap_top_features"] = list(compute_shap_values(model, train_loader, args.model,
                                                                plot_dir=run_dir))[:config.SHAP_MAX_DISPLAY]

    with open(run_dir / "youden_thresholds.json", "w") as f:
        json.dump({args.model: float(threshold)}, f, indent=2)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  {args.model}: Acc={metrics['accuracy']:.4f} Prec={metrics['precision']:.4f} "
          f"Rec={metrics['recall']:.4f} F1={metrics['f1_score']:.4f} AUC={metrics['auc']:.4f}")
    print(f"  Youden J threshold = {threshold:.4f} (J = {j_value:.4f}), "
          f"{metrics['epochs_trained']} epochs")
    print(f"  Everything saved under {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
