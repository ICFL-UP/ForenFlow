"""LSTM inference on sliding windows plus SHAP attributions for each alert."""
import json
import os
import pickle
import warnings

import numpy as np
import torch

import config
from config import FEATURE_COLUMNS
from models.lstm_improved import LSTMImproved

SHAP_VALUES_PATH = config.PLOT_DIR / "shap_values.json"

_shap_cache = None


def load_shap_attributions(shap_path=SHAP_VALUES_PATH):
    global _shap_cache
    if _shap_cache is None and os.path.exists(shap_path):
        with open(shap_path) as f:
            _shap_cache = json.load(f)
    return _shap_cache


def annotate_alert(detection, top_n=5):
    """Attach the global (dataset-wide) SHAP ranking so an alert can be compared to the baseline."""
    shap_vals = load_shap_attributions()
    if shap_vals is None:
        detection["shap_features"] = []
        return detection
    ranked = sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)
    detection["shap_features"] = [
        {"feature": name, "mean_abs_shap": round(val, 6)} for name, val in ranked[:top_n]
    ]
    return detection


def compute_detection_shap(model, window_tensor, background_tensor, top_n=5):
    """Per-detection attribution: gradient of the output w.r.t. the input, scaled by the
    deviation from the background mean (a single-step integrated-gradients approximation)."""
    bg_mean = background_tensor.mean(dim=0, keepdim=True)
    diff = (window_tensor - bg_mean).requires_grad_(True)

    model.zero_grad()
    model(bg_mean + diff).backward()

    attribution = (diff.grad.abs() * diff.detach().abs()).squeeze(0).mean(dim=0).numpy()
    ranked = sorted(zip(FEATURE_COLUMNS, attribution), key=lambda x: x[1], reverse=True)
    return [{"feature": name, "shap_value": round(float(val), 6)}
            for name, val in ranked[:top_n]]


def load_model(model_path=None, scaler_path=None, threshold_path=config.YOUDEN_RESULTS_FILE):
    """Load the trained LSTM Improved, its fitted scaler and the Youden J threshold."""
    model_path = model_path or config.MODEL_DIR / "LSTM_Improved.pt"
    scaler_path = scaler_path or config.MODEL_DIR / "scaler.pkl"

    print(f"  [Model] Loading LSTM model from {model_path} ...")
    model = LSTMImproved(n_features=config.NUM_FEATURES)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    print(f"  [Model] Loading scaler from {scaler_path} ...")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    threshold = None
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold = json.load(f).get("LSTM_Improved")
    if threshold is None:
        warnings.warn(f"WARN: Youden J threshold for LSTM_Improved not found in {threshold_path}")

    load_shap_attributions()
    return model, scaler, threshold


def _score_windows(model, features_scaled, seq_len):
    """Slide a window of seq_len over the scaled flow matrix and score each one."""
    scores = []
    with torch.no_grad():
        for i in range(len(features_scaled) - seq_len + 1):
            tensor = torch.FloatTensor(features_scaled[i:i + seq_len]).unsqueeze(0)
            scores.append(torch.sigmoid(model(tensor)).item())
    return scores


def run_detection(model, scaler, flow_records, threshold, seq_len=config.SEQUENCE_LENGTH):
    """Attack Detection steps 4-7.: scale each flow vector, group the
    scaled vectors into overlapping windows, score them and compare with the threshold."""
    if threshold is None:
        raise ValueError("Youden J threshold is required; refusing to run with a fallback")

    print(f"  [Detect] Running LSTM inference on {len(flow_records)} flows ...")
    if len(flow_records) < seq_len:
        print(f"    Not enough flows (need {seq_len}, have {len(flow_records)})")
        return []

    flow_records.sort(key=lambda r: r["timestamp"])
    features = np.array([[r[col] for col in FEATURE_COLUMNS] for r in flow_records],
                        dtype=np.float32)
    # Scaling before windowing: the scaler works per feature, so every flow is scaled once
    features_scaled = scaler.transform(features)

    detections = []
    for i, confidence in enumerate(_score_windows(model, features_scaled, seq_len)):
        evidence = flow_records[i:i + seq_len]
        detection = {
            "window_start": i,
            "window_end": i + seq_len,
            "confidence": round(confidence, 4),
            "detected": confidence >= threshold,
            "timestamp": evidence[-1]["timestamp"],
            "evidence_flows": [
                {"src_vm": f["src_vm"], "dst_vm": f["dst_vm"],
                 "src_ip": f["src_ip"], "dst_ip": f["dst_ip"],
                 "protocol": f["Protocol"],
                 "packet_rate": round(f["PacketRate"], 2),
                 "packet_size_stddev": round(f["Entropy"], 4)}
                for f in evidence
            ],
        }
        detections.append(detection)

    flagged = [i for i, d in enumerate(detections) if d["detected"]]
    print(f"    {len(flagged)} / {len(detections)} windows flagged as attacks")
    if detections:
        print(f"    Max confidence: {max(d['confidence'] for d in detections):.4f}")
        print(f"    Min confidence: {min(d['confidence'] for d in detections):.4f}")

    if flagged:
        print(f"    Computing per-detection SHAP for {len(flagged)} flagged windows ...")
        # Unflagged windows form the background; fall back to the first windows if all flagged
        background = [i for i in range(len(detections)) if i not in set(flagged)][:50]
        background = background or list(range(min(50, len(detections))))
        bg_tensor = torch.FloatTensor(
            np.array([features_scaled[j:j + seq_len] for j in background]))
        for idx in flagged:
            window = torch.FloatTensor(features_scaled[idx:idx + seq_len]).unsqueeze(0)
            annotate_alert(detections[idx])
            detections[idx]["per_detection_shap"] = compute_detection_shap(model, window, bg_tensor)

    return detections
