#!/usr/bin/env python3
"""Benign baseline: normal inter-VM traffic through the FULL pipeline.

Two modes:
  --diverse   Multi-destination traffic (ICMP pairs + HTTP + DNS) that is more
              likely to trigger ML false positives. The full pipeline runs if
              any window is flagged.
  (default)   Low-rate single-source ICMP for a clean benign test.

Usage on the VM:
    python3 friaas_normal_only.py --diverse --capture-seconds 50
    python3 friaas_normal_only.py --capture-seconds 25 --pps 1 --targets 1
"""
import argparse
import os
import sys
import time

import pipeline
from services.constants import OUTPUT_DIR_NORMAL as OUTPUT_DIR
from services.listener import capture_traffic
from services.ml import load_model
from testbed_traffic import (generate_benign_icmp_load, generate_normal_traffic,
                             start_guest_services, stop_all, ATTACKER_VM)
from services.constants import VM_MAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diverse", action="store_true",
                        help="use multi-destination traffic (all VM pairs + HTTP + DNS)")
    parser.add_argument("--capture-seconds", type=int, default=50,
                        help="capture duration in seconds")
    parser.add_argument("--pps", type=float, default=1,
                        help="pings per second per target (ignored with --diverse)")
    parser.add_argument("--targets", type=int, default=1,
                        help="how many victim VMs to ping (ignored with --diverse)")
    parser.add_argument("--out", default=OUTPUT_DIR)
    args = parser.parse_args()

    mode = "diverse" if args.diverse else "controlled"
    print("=" * 70)
    print("  F-RIaaS Benign Baseline (Full Pipeline)")
    print("  Environment: DevStack 2025.1 / OVS / CirrOS VMs / LSTM Model")
    print(f"  Traffic mode: {mode}")
    print("  Objective: Run the FULL pipeline on benign traffic")
    print("=" * 70)
    os.makedirs(args.out, exist_ok=True)
    timeline = pipeline.Timeline()

    print("\n[1] Loading trained LSTM detection model ...")
    model, scaler, threshold = load_model()
    if threshold is None:
        sys.exit("    ERROR: Youden J threshold missing, run run_all.py first")
    print(f"    Model loaded successfully (threshold={threshold:.4f})")

    start_guest_services()

    pcap_path = os.path.join(args.out, "capture_benign.pcap")
    print(f"\n[2] Starting traffic capture ({args.capture_seconds}s) ...")
    capture_proc = capture_traffic(pcap_path, duration_sec=args.capture_seconds)
    time.sleep(2)

    if args.diverse:
        print("\n[3] Generating DIVERSE normal traffic (all VM pairs + HTTP + DNS) ...")
        normal_procs = generate_normal_traffic(duration_sec=15)
        time.sleep(20)
        normal_procs2 = generate_normal_traffic(duration_sec=10)
        time.sleep(18)
        capture_proc.wait(timeout=30)
        stop_all(normal_procs + normal_procs2)
    else:
        victim_ips = [v["ip"] for name, v in VM_MAP.items() if name != ATTACKER_VM]
        targets = victim_ips[:args.targets]
        print(f"\n[3] Generating benign ICMP traffic ({args.pps} pps to {len(targets)} target(s)) ...")
        procs = generate_benign_icmp_load(args.capture_seconds, args.pps, targets=targets)
        time.sleep(args.capture_seconds)
        stop_all(procs)
        capture_proc.wait(timeout=30)

    time.sleep(2)
    if not os.path.exists(pcap_path):
        sys.exit("    ERROR: No pcap file!")
    print(f"    Captured {os.path.getsize(pcap_path)} bytes")

    flow_records, detections, inference_ms = pipeline.attack_detection(
        model, scaler, threshold, pcap_path, args.out, timeline, suffix="_benign")
    if not detections:
        sys.exit("    ERROR: No detection results!")

    false_positives = [d for d in detections if d["detected"]]
    fp_rate = len(false_positives) / len(detections) * 100

    if not false_positives:
        print(f"\n    All {len(detections)} windows correctly classified as benign")
        print("    No false positives, pipeline not triggered")
        print("\n  BENIGN EXPERIMENT COMPLETE (clean)")
        print(f"  Flows Extracted  : {len(flow_records)}")
        print(f"  Windows Analyzed : {len(detections)}")
        print(f"  False Positives  : 0")
        print(f"  FP Rate          : 0.0%")
        print("  Pipeline         : Not triggered (no detections above threshold)")
        return

    print(f"\n    {len(false_positives)} false positive(s) detected (FP rate {fp_rate:.1f}%)")
    print("    Running the full pipeline on this detection ...")

    alert = pipeline.generate_alert(detections, threshold, timeline)
    if not alert:
        sys.exit("    ERROR: alert generation failed despite false positives")

    response = pipeline.initial_response(alert, pcap_path, timeline)
    analysis = pipeline.incident_response(response, detections, pcap_path, timeline)
    decision = analysis["decision"]

    if decision["verdict"] == "CONFIRMED":
        report, json_path, pdf_path, report_ms = pipeline.reporting(
            alert, response, analysis, detections, threshold,
            pcap_path, args.out, timeline, incident_id=9999)
    else:
        pipeline.dismiss_alert(alert, response, analysis, detections, threshold, pcap_path, timeline)

    print("\n  BENIGN EXPERIMENT COMPLETE")
    print(f"  Flows Extracted  : {len(flow_records)}")
    print(f"  Windows Analyzed : {len(detections)}")
    print(f"  False Positives  : {len(false_positives)}")
    print(f"  FP Rate          : {fp_rate:.1f}%")
    print(f"  Verdict          : {decision['verdict']} ({decision['sources_flagged']}/3 sources)")
    print(f"  ML Detected      : {decision['ml_detected']}")
    print(f"  Logs Suspicious  : {decision['logs_suspicious']}")
    print(f"  PCAP Suspicious  : {decision['pcap_suspicious']}")
    if decision["verdict"] != "CONFIRMED":
        print("  Action           : VM restored, verdict logged")
    else:
        print("  Action           : Report generated (false positive CONFIRMED)")


if __name__ == "__main__":
    main()
