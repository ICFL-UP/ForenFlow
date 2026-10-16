#!/usr/bin/env python3
"""Capture real testbed traffic and produce dataset3_testbed.csv (training source 3).

Usage:  python3 capture_dataset3.py [--rounds 3] [--attack-rounds N] [--duration 60]
                                    [--output data/raw/dataset3_testbed.csv]
A longer capture gives more flow rows, for example for the window-length ablation. Normal
rounds overlap in time (each round pings for duration/rounds seconds), so one normal round
keeps benign traffic flowing for the whole phase, while attack rounds run back to back and
so scale the number of attack flows.
"""
import argparse
import os
import subprocess
import sys
import time

import pandas as pd

import config
from services.constants import MIRROR_IFACE
from services.listener import _extract_tcpdump
from testbed_traffic import (generate_normal_traffic, generate_attack_traffic, start_guest_services,
                             stop_all)

OUTPUT_DIR = os.path.expanduser("~/dataset3_capture")
PCAP_NORMAL = os.path.join(OUTPUT_DIR, "normal.pcap")
PCAP_ATTACK = os.path.join(OUTPUT_DIR, "attack.pcap")
CSV_OUTPUT = config.RAW_DIR / "dataset3_testbed.csv"


def start_capture(pcap_path):
    proc = subprocess.Popen(f"sudo tcpdump -i {MIRROR_IFACE} -w {pcap_path} -c 100000",
                            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    return proc


def capture_phase(pcap_path, traffic_fn, duration_sec=60):
    capture = start_capture(pcap_path)
    procs = traffic_fn()
    time.sleep(duration_sec + 5)
    stop_all(procs)
    capture.kill()
    time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3, help="normal traffic rounds")
    parser.add_argument("--attack-rounds", type=int, default=None,
                        help="attack traffic rounds (defaults to --rounds)")
    parser.add_argument("--duration", type=int, default=60, help="seconds per phase")
    parser.add_argument("--output", default=str(CSV_OUTPUT))
    args = parser.parse_args()
    csv_output = args.output
    attack_rounds = args.attack_rounds or args.rounds

    print("=" * 60)
    print("  Dataset 3 - Real Testbed Traffic Capture")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_guest_services()

    print(f"\n[Phase 1] Capturing NORMAL traffic ({args.duration}s) ...")
    capture_phase(PCAP_NORMAL, lambda: generate_normal_traffic(
        duration_sec=args.duration, rounds=args.rounds), duration_sec=args.duration)

    print(f"\n[Phase 2] Capturing ATTACK traffic ({args.duration}s) ...")
    capture_phase(PCAP_ATTACK, lambda: generate_attack_traffic(
        rounds=attack_rounds, ports=(22, 80, 443, 3306, 8080, 5000, 8443, 9090),
        ping_count=100, connections_per_port=30, exfil_kb=200), duration_sec=args.duration)

    print("\n[Phase 3] Extracting features ...")
    # two packets are enough for a training flow; live detection keeps the stricter minimum
    normal_records = _extract_tcpdump(PCAP_NORMAL, max_lines=100000, min_packets=2)
    attack_records = _extract_tcpdump(PCAP_ATTACK, max_lines=100000, min_packets=2)
    print(f"  Normal flows: {len(normal_records)}")
    print(f"  Attack flows: {len(attack_records)}")
    if not normal_records and not attack_records:
        sys.exit("ERROR: No flow records extracted from either capture!")

    for record in normal_records:
        record[config.LABEL_COLUMN] = 0
    for record in attack_records:
        record[config.LABEL_COLUMN] = 1

    df = pd.DataFrame(normal_records + attack_records)
    df = df[config.FEATURE_COLUMNS + [config.LABEL_COLUMN]]
    df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    df.to_csv(csv_output, index=False)

    print(f"\n  Dataset 3 saved -> {csv_output}")
    print(f"  Total rows: {len(df)}")
    print(f"  Normal: {len(normal_records)}, Attack: {len(attack_records)}")
    print(f"  Class balance: {len(attack_records) / len(df) * 100:.1f}% attack")


if __name__ == "__main__":
    main()
