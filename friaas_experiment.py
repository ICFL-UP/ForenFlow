#!/usr/bin/env python3
"""Attack scenario: normal traffic, a flood from vm-app-02, then the full four-phase pipeline."""
import json
import os
import sys
import time

import numpy as np

import subprocess

import pipeline
from services.constants import (VM_MAP, OUTPUT_DIR_ATTACK as OUTPUT_DIR,
                                DHCP_NS, GUEST_USER, GUEST_PASSWORD)
from services.listener import capture_traffic
from services.ml import load_model
from services.openstack_service import restore_vm
from testbed_traffic import (generate_normal_traffic, generate_attack_traffic, start_guest_services,
                             stop_all)


def run_capture_with_attack(pcap_path, timeline):
    """One 70 s capture segment: 15 s normal, 20 s attack, 15 s normal."""
    start_guest_services()
    print("\n[2] Starting traffic capture on OVS mirror port ...")
    capture_proc = capture_traffic(pcap_path, duration_sec=70)
    timeline.add("Listener", "Traffic capture started on mirror0 (70s window)")
    time.sleep(2)

    print("\n[3] Phase 1: Generating NORMAL traffic (15s) ...")
    normal_procs = generate_normal_traffic(duration_sec=15)
    timeline.add("Listener", "Normal inter-VM traffic generation started (Phase 1)")
    time.sleep(20)

    print("\n[4] Phase 2: Generating ATTACK traffic (20s) ...")
    attacker_ip = VM_MAP["vm-app-02"]["ip"]
    subprocess.Popen(
        f"sudo ip netns exec {DHCP_NS} sshpass -p {GUEST_PASSWORD} "
        f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "
        f"{GUEST_USER}@{attacker_ip} "
        f"'sudo nohup tcpdump -i eth0 -w /tmp/sniff.pcap -C 1 -W 1 >/dev/null 2>&1 &'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    attack_procs = generate_attack_traffic()
    timeline.add("Listener", "SYN flood attack traffic started (Phase 2)")
    time.sleep(20)

    print("\n[5] Phase 3: Resuming NORMAL traffic (15s) ...")
    normal_procs2 = generate_normal_traffic(duration_sec=10)
    timeline.add("Listener", "Normal traffic resumed (Phase 3)")
    time.sleep(15)

    print("\n[6] Waiting for capture to complete ...")
    capture_proc.wait(timeout=30)
    stop_all(normal_procs + attack_procs + normal_procs2)
    time.sleep(2)

    if not os.path.exists(pcap_path):
        print("    ERROR: No pcap file captured!")
        sys.exit(1)
    print(f"    Captured {os.path.getsize(pcap_path)} bytes in {pcap_path}")


def main():
    print("=" * 70)
    print("  F-RIaaS Real OpenStack Detection Experiment")
    print("  Environment: DevStack 2025.1 / OVS / CirrOS VMs / LSTM Model")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timeline = pipeline.Timeline()

    print("\n[1] Loading trained LSTM detection model ...")
    model, scaler, threshold = load_model()
    if threshold is None:
        sys.exit("    ERROR: Youden J threshold missing, run run_all.py first")
    print(f"    Model loaded successfully (threshold={threshold:.4f})")

    pcap_path = os.path.join(OUTPUT_DIR, "capture.pcap")
    run_capture_with_attack(pcap_path, timeline)

    flow_records, detections, inference_ms = pipeline.attack_detection(
        model, scaler, threshold, pcap_path, OUTPUT_DIR, timeline)
    if not detections:
        sys.exit("    No detection results (insufficient flows)")

    alert = pipeline.generate_alert(detections, threshold, timeline)
    response = analysis = None
    decision = {"verdict": "BENIGN", "summary": "Verdict: BENIGN (0/3 sources flagged)"}

    if alert:
        response = pipeline.initial_response(alert, pcap_path, timeline)
        analysis = pipeline.incident_response(response, detections, pcap_path, timeline)
        decision = analysis["decision"]

        if decision["verdict"] == "CONFIRMED":
            report, json_path, pdf_path, report_ms = pipeline.reporting(
                alert, response, analysis, detections, threshold, pcap_path, OUTPUT_DIR,
                timeline, incident_id=1)
            latencies = pipeline.collect_stage_latencies(inference_ms, response, analysis, report_ms)

            # PoC restores the VM after evidence collection so the testbed can be reused.
            # In production the VM would remain quarantined until an administrator releases it.
            if response["isolation"]:
                print(f"\n[14] Restoring {response['vm_name']} (removing isolation) ...")
                restore_vm(response["vm_name"], response["isolation"]["security_group"],
                           response["isolation"].get("original_security_groups"))
        else:
            pipeline.dismiss_alert(alert, response, analysis, detections, threshold,
                                   pcap_path, timeline)
            latencies = pipeline.collect_stage_latencies(inference_ms, response, analysis)

        with open(os.path.join(OUTPUT_DIR, "pipeline_timings.json"), "w") as f:
            json.dump(latencies, f, indent=2)

    attacker_ip = VM_MAP["vm-app-02"]["ip"]
    subprocess.run(
        f"sudo ip netns exec {DHCP_NS} sshpass -p {GUEST_PASSWORD} "
        f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "
        f"{GUEST_USER}@{attacker_ip} 'sudo killall tcpdump 2>/dev/null'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=15)

    attack_detections = [d for d in detections if d["detected"]]
    confirmed = decision["verdict"] == "CONFIRMED"
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\n  Environment      : OpenStack DevStack 2025.1 (VirtualBox)")
    print("  Networking       : Neutron + Open vSwitch")
    print("  Traffic Capture  : OVS Port Mirror on br-int")
    print(f"  VMs              : {len(VM_MAP)} CirrOS instances")
    print(f"  Flows Extracted  : {len(flow_records)}")
    print(f"  Windows Analyzed : {len(detections)}")
    print(f"  Attacks Detected : {len(attack_detections)}")
    print(f"  Detection Rate   : {len(attack_detections) / len(detections) * 100:.1f}% of windows")
    if attack_detections:
        confidences = [d["confidence"] for d in attack_detections]
        print(f"  Max Confidence   : {max(confidences):.4f}")
        print(f"  Avg Confidence   : {np.mean(confidences):.4f}")
    print(f"  Verdict          : {decision['verdict']}")
    print(f"  Action           : {'Full forensic report generated' if confirmed else 'Verdict logged (dismissed)'}")
    print(f"  Isolation        : {'Yes' if response and response['isolation'] else 'No'}")
    print(f"\n  Outputs -> {OUTPUT_DIR}/")
    print("    - capture.pcap")
    print("    - extracted_features.json")
    print("    - detection_results.json")
    if confirmed:
        print("    - forensic_report.json / forensic_report.pdf")
    if response:
        print(f"  Evidence -> {response['case_dir']}/")
    print()


if __name__ == "__main__":
    main()
