"""False positive rate under benign ICMP load: idle, low-rate, high-rate.

Each scenario pushes ICMP echo from the DHCP namespace to the three victim VMs at a fixed
rate for --duration seconds, captures the mirror, and counts flagged windows. The response
phases are not run: the point is to measure the detector, not to quarantine a benign VM.

Usage on the VM:  python3 fp_load.py --pps 0 1 150 --duration 600
"""
import argparse
import json
import os
import sys
import time

import pipeline
from services.constants import OUTPUT_DIR_NORMAL
from services.listener import capture_traffic
from services.ml import load_model
from testbed_traffic import generate_benign_icmp_load, stop_all


def run_scenario(pps, duration, model, scaler, threshold, out_dir):
    label = f"{pps}pps"
    print("\n" + "#" * 70)
    print(f"  BENIGN LOAD: {pps} ping(s)/s per target for {duration}s")
    print("#" * 70)
    timeline = pipeline.Timeline()
    pcap_path = os.path.join(out_dir, f"capture_{label}.pcap")

    capture_proc = capture_traffic(pcap_path, duration_sec=duration, max_packets=10_000_000)
    time.sleep(2)
    procs = generate_benign_icmp_load(duration, pps)
    time.sleep(duration)
    stop_all(procs)
    capture_proc.wait(timeout=60)

    flows, detections, inference_ms = pipeline.attack_detection(
        model, scaler, threshold, pcap_path, out_dir, timeline, suffix=f"_{label}")
    false_positives = [d for d in detections if d["detected"]]
    fp_rate = len(false_positives) / len(detections) if detections else 0.0
    print(f"    {len(false_positives)}/{len(detections)} windows flagged -> FP rate {fp_rate * 100:.2f}%")
    return {
        "pings_per_second_per_target": pps,
        "duration_seconds": duration,
        "pcap_bytes": os.path.getsize(pcap_path) if os.path.exists(pcap_path) else 0,
        "flows_emitted": len(flows),
        "predictions": len(detections),
        "false_positive_detections": len(false_positives),
        "fp_rate": round(fp_rate, 4),
        "max_confidence": max((d["confidence"] for d in detections), default=0.0),
        "inference_ms": inference_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pps", type=float, nargs="+", default=[0, 1, 150],
                        help="pings per second per target, one scenario each")
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR_NORMAL, "fp_load"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, scaler, threshold = load_model()
    if threshold is None:
        sys.exit("Youden J threshold missing, run run_all.py first")

    results = [run_scenario(pps, args.duration, model, scaler, threshold, args.out)
               for pps in args.pps]
    path = os.path.join(args.out, "fp_load.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  FALSE POSITIVE LOAD CURVE")
    print("=" * 70)
    print(f"  {'pps':>6} {'predictions':>12} {'FP':>6} {'FP rate':>9}")
    for r in results:
        print(f"  {r['pings_per_second_per_target']:>6} {r['predictions']:>12} "
              f"{r['false_positive_detections']:>6} {r['fp_rate'] * 100:>8.2f}%")
    print(f"\n  Results -> {path}")


if __name__ == "__main__":
    main()
