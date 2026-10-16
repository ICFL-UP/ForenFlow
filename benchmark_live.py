"""Repeated live detection benchmark with per-stage latencies.

Each iteration runs a SYN flood from vm-app-02 in a background thread so the flood
stays active through detection and isolation, matching the real attack condition.
Aggregates land in repeat_runs.json.

Usage on the VM:  python3 benchmark_live.py --iterations 15 --capture-seconds 40 --flood-rounds 8
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import pipeline
from services.constants import OUTPUT_DIR_ATTACK
from services.listener import capture_traffic
from services.ml import load_model
from services.openstack_service import restore_vm
from testbed_traffic import generate_attack_traffic, start_guest_services, stop_all


def _spawn_flood(procs_out, rounds, stop_event):
    """Run attack traffic in a background thread so it persists through detection."""
    for _ in range(rounds):
        if stop_event.is_set():
            break
        procs = generate_attack_traffic(
            rounds=1, ping_count=200, connections_per_port=50, exfil_kb=500)
        procs_out.extend(procs)
        time.sleep(3)


def run_iteration(i, model, scaler, threshold, args, out_dir):
    print("\n" + "#" * 70)
    print(f"  ITERATION {i}/{args.iterations}")
    print("#" * 70)
    timeline = pipeline.Timeline()
    pcap_path = os.path.join(out_dir, f"capture_iter{i:02d}.pcap")

    flood_procs = []
    stop_event = threading.Event()
    flood_thread = threading.Thread(
        target=_spawn_flood, args=(flood_procs, args.flood_rounds, stop_event), daemon=True)
    flood_thread.start()
    time.sleep(8)

    flood_start = time.time()
    capture_proc = capture_traffic(pcap_path, duration_sec=args.capture_seconds)
    capture_proc.wait(timeout=args.capture_seconds + 30)

    flood_alive_at_detection = any(p.poll() is None for p in flood_procs)
    print(f"  Flood alive at detection: {flood_alive_at_detection}")

    result = {"iteration": i, "flood_start_utc_epoch": flood_start}
    flows, detections, inference_ms = pipeline.attack_detection(
        model, scaler, threshold, pcap_path, out_dir, timeline, suffix=f"_iter{i:02d}")
    flagged = [d for d in detections if d["detected"]]
    result.update({
        "flows_emitted": len(flows),
        "total_predictions": len(detections),
        "attack_detections": len(flagged),
        "attack_rate": round(len(flagged) / len(detections), 4) if detections else 0.0,
        "max_confidence": max((d["confidence"] for d in detections), default=0.0),
        "inference_ms": inference_ms,
        "flood_alive_at_detection": flood_alive_at_detection,
    })

    alert = pipeline.generate_alert(detections, threshold, timeline)
    if not alert:
        stop_event.set()
        stop_all(flood_procs)
        result.update({"detected": False, "verdict": None,
                       "flood_alive_at_isolation": None})
        return result

    flood_alive_at_isolation = any(p.poll() is None for p in flood_procs)
    print(f"  Flood alive at isolation: {flood_alive_at_isolation}")

    response = pipeline.initial_response(alert, pcap_path, timeline)
    analysis = pipeline.incident_response(response, detections, pcap_path, timeline)
    decision = analysis["decision"]
    report_ms = None
    if decision["verdict"] == "CONFIRMED":
        *_, report_ms = pipeline.reporting(alert, response, analysis, detections, threshold,
                                           pcap_path, out_dir, timeline, incident_id=i)
    else:
        pipeline.dismiss_alert(alert, response, analysis, detections, threshold, pcap_path, timeline)

    stop_event.set()
    stop_all(flood_procs)

    if response["isolation"] and decision["verdict"] == "CONFIRMED":
        restore_vm(response["vm_name"], response["isolation"]["security_group"],
                   response["isolation"].get("original_security_groups"))

    result.update({
        "detected": True,
        "suspect_vm": alert["suspect_vm"],
        "verdict": decision["verdict"],
        "sources_flagged": decision["sources_flagged"],
        "case_confidence": alert["score"],
        "case_dir": response["case_dir"],
        "isolated": response["isolation"] is not None,
        "flood_alive_at_isolation": flood_alive_at_isolation,
        "stage_latencies_ms": pipeline.collect_stage_latencies(
            inference_ms, response, analysis, report_ms),
    })
    return result


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {"mean": round(statistics.mean(values), 3),
            "std": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def summarise(results):
    detected = [r for r in results if r["detected"]]
    stage_names = ("inference_ms", "isolation_ms", "preservation_ms", "correlation_ms", "report_ms")
    flood_alive_count = sum(1 for r in results if r.get("flood_alive_at_isolation"))
    return {
        "iterations": len(results),
        "attack_detection_rate": round(len(detected) / len(results), 4) if results else 0.0,
        "flood_alive_at_isolation": f"{flood_alive_count}/{len(results)}",
        "predictions_per_iteration": _mean_std([r["total_predictions"] for r in results]),
        "flows_per_iteration": _mean_std([r["flows_emitted"] for r in results]),
        "cases_generated": len(detected),
        "confirmed_cases": sum(1 for r in detected if r["verdict"] == "CONFIRMED"),
        "sources_flagged": sorted({r["sources_flagged"] for r in detected}),
        "case_confidence": _mean_std([r["case_confidence"] for r in detected]),
        "stage_latencies_ms": {
            stage: _mean_std([r["stage_latencies_ms"].get(stage) for r in detected])
            for stage in stage_names
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--capture-seconds", type=int, default=40)
    parser.add_argument("--flood-rounds", type=int, default=8,
                        help="attack traffic rounds (each ~15s, must outlast capture + detection)")
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR_ATTACK, "benchmark"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, scaler, threshold = load_model()
    if threshold is None:
        sys.exit("Youden J threshold missing, run run_all.py first")
    start_guest_services()

    results = [run_iteration(i, model, scaler, threshold, args, args.out)
               for i in range(1, args.iterations + 1)]

    flood_alive_count = sum(1 for r in results if r.get("flood_alive_at_isolation"))
    payload = {
        "iterations": args.iterations,
        "capture_seconds": args.capture_seconds,
        "flood_rounds": args.flood_rounds,
        "results": results,
        "summary": summarise(results),
    }
    path = os.path.join(args.out, "repeat_runs.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)
    print(json.dumps(payload["summary"], indent=2))
    print(f"  Flood alive at isolation: {flood_alive_count}/{len(results)}")
    print(f"\n  Results -> {path}")


if __name__ == "__main__":
    main()
