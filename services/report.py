"""Assemble the forensic report dictionary the PDF is rendered from it."""
import hashlib
import json
import os
from datetime import datetime, timezone

import config


def _sha256_file(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def generate_forensic_report(detections, pcap_path, incident_id, threshold,
                             isolation=None, preservation=None, correlation=None,
                             infra_logs=None, shap_features=None, timeline_events=None,
                             suspect=None):
    attack_detections = [d for d in detections if d["detected"]]
    affected_vms = {vm for d in attack_detections
                    for flow in d["evidence_flows"] for vm in (flow["src_vm"], flow["dst_vm"])}
    is_attack = bool(attack_detections)

    report = {
        "report_id": f"FR-{incident_id:04d}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "PACKET_SNIFFING_DETECTED" if is_attack else "NO_THREAT_DETECTED",
        "severity": "HIGH" if is_attack else "LOW",
        "environment": "OpenStack DevStack (Real VMs via VirtualBox)",
        "incident_details": {
            "incident_id": incident_id,
            "total_windows_analyzed": len(detections),
            "attack_windows_detected": len(attack_detections),
            "max_confidence": max((d["confidence"] for d in detections), default=0),
            "affected_vms": sorted(affected_vms),
            "suspect_vm": suspect,
            "detection_timeline": {
                "first_detection": attack_detections[0]["timestamp"] if attack_detections else None,
                "last_detection": attack_detections[-1]["timestamp"] if attack_detections else None,
            },
        },
        "detection_method": {
            "model": "LSTM_Improved",
            "architecture": "LSTM(128)->LSTM(64)->Dense(32)->Dense(1)",
            "pooling": "mean over all time steps",
            "input_sequence_length": config.SEQUENCE_LENGTH,
            "threshold": threshold,
            "feature_count": config.NUM_FEATURES,
            "deployment": "Real-time OVS mirror capture on OpenStack",
        },
        "isolation_action": {"performed": isolation is not None, "details": isolation},
        "chain_of_custody": {
            "evidence_hash_sha256": hashlib.sha256(
                json.dumps(detections, sort_keys=True, default=str).encode()).hexdigest(),
            "pcap_hash_sha256": _sha256_file(pcap_path),
            "pcap_file": pcap_path,
            "integrity_verified": bool(preservation and preservation.get("chain_verified")
                                       and preservation.get("content_verified")),
            "collection_method": "OVS port mirror on br-int (real traffic)",
            "storage": (preservation or {}).get("evidence_dir", "Local forensic evidence store"),
        },
        "detection_breakdown": [
            {"window_start": d["window_start"], "window_end": d["window_end"],
             "confidence": d["confidence"], "timestamp": d["timestamp"],
             "per_detection_shap": d.get("per_detection_shap", [])}
            for d in attack_detections
        ],
    }

    if preservation:
        report["evidence_preservation"] = preservation
    if correlation:
        report["correlation_analysis"] = correlation
    if infra_logs:
        report["infrastructure_logs"] = infra_logs
    if shap_features:
        report["shap_attributions"] = shap_features
    if timeline_events:
        ordered = sorted(timeline_events, key=lambda e: e.get("timestamp", ""))
        for i, event in enumerate(ordered, 1):
            event["sequence"] = i
        report["chronological_timeline"] = ordered
    return report
