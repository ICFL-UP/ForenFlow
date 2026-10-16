"""Verdict Logging: the audit record of a dismissed alert, chained by SHA-256."""
import hashlib
import json
import os
from datetime import datetime, timezone

from services.constants import EVIDENCE_DIR

VERDICT_LOG_FILE = os.path.join(EVIDENCE_DIR, "verdict_log.jsonl")


def _last_entry_hash():
    if not os.path.exists(VERDICT_LOG_FILE):
        return ""
    last_line = ""
    with open(VERDICT_LOG_FILE) as f:
        for line in f:
            if line.strip():
                last_line = line
    try:
        return json.loads(last_line).get("entry_hash", "") if last_line else ""
    except json.JSONDecodeError:
        return ""


def _hash_file(path):
    if not path or not os.path.exists(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_data(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def log_verdict(case_id, decision, detections, threshold, log_analysis=None,
                pcap_analysis=None, isolation=None, pcap_path=None,
                preservation=None, alert=None, log_excerpts=None):
    """Append one chained entry holding everything the model says a dismissal must keep:
    score and window, SHAP, log excerpts, the five indicator outcomes, PCAP findings, the
    reasoning, and hashes that tie the entry back to the repository artefacts."""
    flagged = [d for d in detections if d["detected"]]
    indicator_results = (log_analysis or {}).get("indicator_results", {})

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "verdict": decision["verdict"],
        "detection_summary": {
            "total_windows": len(detections),
            "flagged_windows": len(flagged),
            "threshold": threshold,
            "max_score": max((d["confidence"] for d in detections), default=0),
        },
        "flagged_window_details": [
            {"window_start": d["window_start"], "window_end": d["window_end"],
             "confidence": d["confidence"], "timestamp": d.get("timestamp", "")}
            for d in flagged
        ],
        "shap_attributions": (alert or {}).get("shap_features", []),
        "per_detection_shap": [d.get("per_detection_shap", []) for d in flagged],
        "evidence_sources": {
            "ml_detected": decision["ml_detected"],
            "logs_suspicious": decision["logs_suspicious"],
            "pcap_suspicious": decision["pcap_suspicious"],
            "evidence_count": decision["sources_flagged"],
        },
        "indicator_outcomes": {name: result["flagged"]
                               for name, result in indicator_results.items()},
        "indicator_details": {name: result["detail"]
                              for name, result in indicator_results.items()},
        "log_excerpts": log_excerpts or {},
        "log_analysis_summary": (log_analysis or {}).get("summary", "Not performed"),
        "pcap_findings": (pcap_analysis or {}).get("findings", []),
        "reasoning": decision["summary"],
        "action_taken": "VM isolated then restored" if isolation else "No isolation performed",
        "isolation_details": isolation,
        "evidence_dir": (preservation or {}).get("evidence_dir"),
        "artefact_hashes": {
            "pcap_sha256": _hash_file(pcap_path),
            "detections_sha256": _hash_data(detections),
            "log_analysis_sha256": _hash_data(log_analysis) if log_analysis else "",
            "pcap_analysis_sha256": _hash_data(pcap_analysis) if pcap_analysis else "",
            "chain_of_custody_last_hash": (
                preservation["chain_of_custody"][-1]["chained_hash"]
                if preservation and preservation.get("chain_of_custody") else ""
            ),
        },
    }

    prev_hash = _last_entry_hash()
    entry["previous_hash"] = prev_hash
    entry["entry_hash"] = hashlib.sha256(
        (prev_hash + json.dumps(entry, sort_keys=True, default=str)).encode()).hexdigest()

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(VERDICT_LOG_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return {"entry_hash": entry["entry_hash"], "file_path": VERDICT_LOG_FILE, "entry": entry}
