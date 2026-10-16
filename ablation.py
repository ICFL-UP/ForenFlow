#!/usr/bin/env python3
"""Sliding window length ablation : retrain LSTM Improved for each L.

Runs on the testbed capture by default, because the purpose is to compare window lengths
against the traffic of the deployment rather than to maximise absolute scores.

Usage:  python3 ablation.py [--dataset testbed|combined] [--windows 5 10 15 20 30]
                            [--data-file path/to/capture.csv]
--data-file points at a different capture in the 18-feature schema, for example a longer testbed
capture taken only for this study, without replacing training source 3.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import config
from dataset import load_dataframe, make_loader, scale_splits, split_dataframe
from evaluate import predict_scores, youden_threshold
from models.lstm_improved import LSTMImproved
from train import train_model

DEFAULT_WINDOWS = (5, 10, 15, 20, 30)


def run_ablation(dataset="testbed", windows=DEFAULT_WINDOWS, epochs=config.NUM_EPOCHS,
                 data_file=None):
    print("=" * 70)
    print("  SLIDING WINDOW LENGTH ABLATION STUDY")
    print("=" * 70)
    if data_file:
        path = Path(data_file)
    else:
        path = (config.RAW_DIR / "dataset3_testbed.csv" if dataset == "testbed"
                else config.PROCESSED_DIR / "combined_dataset.csv")
    print(f"  Data: {path}")
    train_df, val_df, test_df = split_dataframe(load_dataframe(path))
    # one split and one scaler for every L, so only the window length changes
    _, X_train, X_val, X_test = scale_splits(train_df, val_df, test_df)
    y_train, y_val, y_test = (part[config.LABEL_COLUMN].values for part in (train_df, val_df, test_df))

    results = []
    for L in windows:
        print(f"\n--- Window Length L = {L} ---")
        if min(len(X_train), len(X_val), len(X_test)) < L:
            print(f"  SKIPPED: not enough rows for L={L}")
            continue
        train_loader = make_loader(X_train, y_train, seq_len=L, shuffle=True)
        val_loader = make_loader(X_val, y_val, seq_len=L)
        test_loader = make_loader(X_test, y_test, seq_len=L)
        print(f"  Sequences: train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
              f"test={len(test_loader.dataset)}")

        history, model = train_model(LSTMImproved(), train_loader, val_loader, epochs=epochs,
                                     model_name=f"ablation_L{L}",
                                     save_dir=config.RESULTS_DIR / "ablation")
        val_scores, val_true = predict_scores(model, val_loader)
        threshold, j_value, *_ = youden_threshold(val_true, val_scores)

        test_scores, test_true = predict_scores(model, test_loader)
        y_pred = (test_scores >= threshold).astype(int)
        results.append({
            "window_length": L,
            "accuracy": accuracy_score(test_true, y_pred),
            "precision": precision_score(test_true, y_pred, zero_division=0),
            "recall": recall_score(test_true, y_pred, zero_division=0),
            "f1_score": f1_score(test_true, y_pred, zero_division=0),
            "auc": roc_auc_score(test_true, test_scores) if len(set(test_true)) > 1 else 0.0,
            "youden_j": j_value,
            "threshold": threshold,
            "epochs_trained": len(history["train_loss"]),
            "n_train_seq": len(train_loader.dataset),
            "n_test_seq": len(test_loader.dataset),
        })
        r = results[-1]
        print(f"  Youden J = {j_value:.4f} at T = {threshold:.4f}; "
              f"Acc={r['accuracy']:.4f} F1={r['f1_score']:.4f} AUC={r['auc']:.4f} "
              f"epochs={r['epochs_trained']}")

    print(f"\n{'=' * 70}\n  ABLATION RESULTS SUMMARY\n{'=' * 70}")
    print(f"  {'L':>4} {'Acc':>8} {'F1':>8} {'AUC':>8} {'J':>8} {'Epochs':>7}")
    for r in results:
        print(f"  {r['window_length']:>4} {r['accuracy']:>8.4f} {r['f1_score']:>8.4f} "
              f"{r['auc']:>8.4f} {r['youden_j']:>8.4f} {r['epochs_trained']:>7}")
    if results:
        best = max(results, key=lambda r: r["youden_j"])
        print(f"\n  Highest Youden J: L = {best['window_length']} (J = {best['youden_j']:.4f})")

    out = config.RESULTS_DIR / f"ablation_results_{dataset}.json"
    with open(out, "w") as f:
        json.dump({"dataset": dataset, "data_file": str(path), "n_rows": len(train_df) + len(val_df) + len(test_df),
                   "results": results}, f, indent=2)
    print(f"  Results saved -> {out}")
    if len(results) >= 2:
        plot_ablation(results)
    return results


def plot_ablation(results):
    Ls = [r["window_length"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key, style, label in (("f1_score", "bo-", "F1-Score"), ("auc", "rs--", "AUC"),
                              ("accuracy", "g^:", "Accuracy")):
        axes[0].plot(Ls, [r[key] for r in results], style, label=label, linewidth=2, markersize=8)
    axes[0].set_title("Detection Performance vs Window Length", fontsize=13)
    best_L = max(results, key=lambda r: r["youden_j"])["window_length"]
    axes[0].axvline(best_L, color="blue", alpha=0.2, linewidth=8)

    for key, style, label in (("precision", "mo-", "Precision"), ("recall", "cD--", "Recall")):
        axes[1].plot(Ls, [r[key] for r in results], style, label=label, linewidth=2, markersize=8)
    axes[1].set_title("Precision-Recall vs Window Length", fontsize=13)

    for ax in axes:
        ax.set_xlabel("Window Length (L)", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xticks(Ls)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = config.PLOT_DIR / "ablation_window_length.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Ablation plot saved -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["testbed", "combined"], default="testbed")
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--data-file", default=None)
    args = parser.parse_args()
    run_ablation(args.dataset, args.windows, args.epochs, args.data_file)
