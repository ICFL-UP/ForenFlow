"""The four F-RIaaS phases as functions shared by every scenario script.

    attack_detection()   extract flows, scale, window, infer, threshold
    generate_alert()     alert record and suspect VM identification
    initial_response()   volatile state, quarantine, preserve PCAP and disk
    incident_response()  log collection, PCAP analysis, correlation, verdict
    dismiss_alert()      verdict logging, then restore
    reporting()          timeline, hash chain, PDF report, notification
"""
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone

from generate_further_report_pdf import generate_report_pdf
from services.analysis import analyse_logs, analyse_pcap, correlate_and_decide
from services.constants import VM_MAP
from services.listener import extract_features_from_pcap
from services.ml import run_detection
from services.notifications import notify_incident
from services.openstack_service import (identify_vm_by_ip, isolate_vm, restore_vm,
                                        collect_infrastructure_logs)
from services.preservation import (ChainedHashLog, collect_layer1, collect_layer2_ssh,
                                   check_capture_tools, preserve_disk_and_pcap, new_case_dir)
from services.report import generate_forensic_report
from services.verdict_log import log_verdict


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Timeline:

    def __init__(self):
        self.events = []

    def add(self, source, event):
        self.events.append({"timestamp": utc_now(), "source": source, "event": event})


def _save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def _elapsed_ms(t0):
    return round((time.perf_counter() - t0) * 1000, 2)


# ---------------------------------------------------------------- Attack Detection

def attack_detection(model, scaler, threshold, pcap_path, output_dir, timeline, suffix=""):
    """Run the detection pipeline over one capture segment. Returns (flows, detections, ms)."""
    print("\n[7] Extracting flow features from captured traffic ...")
    flow_records = extract_features_from_pcap(pcap_path)
    if not flow_records:
        print("    ERROR: No flow records extracted!")
        return [], [], 0.0
    _save_json(flow_records, os.path.join(output_dir, f"extracted_features{suffix}.json"))
    timeline.add("Listener", f"Extracted {len(flow_records)} flow records from "
                             f"{os.path.basename(pcap_path)}")

    print("\n[8] Running LSTM detection on captured traffic ...")
    t0 = time.perf_counter()
    detections = run_detection(model, scaler, flow_records, threshold)
    inference_ms = _elapsed_ms(t0)
    _save_json(detections, os.path.join(output_dir, f"detection_results{suffix}.json"))
    return flow_records, detections, inference_ms


# ---------------------------------------------------------------- 

def generate_alert(detections, threshold, timeline):
    """Generate Internal Alert and Identify Suspect VM. Detection and alert share one
    timestamp. Returns None when no window crossed the threshold."""
    print("\n[9] Evaluating detection results ...")
    flagged = [d for d in detections if d["detected"]]
    if not flagged:
        print("    No attacks detected in this capture window")
        return None

    ip_counts = Counter(flow["src_ip"] for d in flagged for flow in d["evidence_flows"])
    suspect_ip, evidence_flows = ip_counts.most_common(1)[0]
    suspect_vm, port_mac = identify_vm_by_ip(suspect_ip)
    suspect_mac = port_mac or VM_MAP.get(suspect_vm, {}).get("mac", "unknown")
    max_confidence = max(d["confidence"] for d in detections)

    alert = {
        "alert_time": utc_now(),
        "score": max_confidence,
        "threshold": threshold,
        "flagged_windows": [d["window_start"] for d in flagged],
        "windows_analysed": len(detections),
        "first_flagged_flow_timestamp": flagged[0]["timestamp"],
        "suspect_ip": suspect_ip,
        "suspect_mac": suspect_mac,
        "suspect_vm": suspect_vm,
        "evidence_flows": evidence_flows,
        "shap_features": flagged[0].get("shap_features", []),
        "per_detection_shap": flagged[0].get("per_detection_shap", []),
    }
    print("    ALERT: Packet sniffing detected!")
    print(f"    Most suspicious VM: {suspect_vm} (IP={suspect_ip}, MAC={suspect_mac}, "
          f"{evidence_flows} evidence flows)")
    timeline.add("ML Detection", f"LSTM Improved flagged {len(flagged)}/{len(detections)} "
                                 f"windows (max confidence {max_confidence:.4f})")
    timeline.add("ML Detection", f"{suspect_vm} identified as most suspicious "
                                 f"(IP={suspect_ip}, MAC={suspect_mac}, {evidence_flows} evidence flows)")
    return alert


# ---------------------------------------------------------------- 

def initial_response(alert, pcap_path, timeline):
    """Collect Volatile State, Apply Quarantine, Preserve PCAP and Disk State, in the order in
    which the evidence would otherwise disappear."""
    vm_name = alert["suspect_vm"]
    hash_log = ChainedHashLog()
    latencies = {}

    print("\n[9b] Collecting volatile state and disk artefacts before isolation ...")
    t0 = time.perf_counter()
    layer1 = collect_layer1(vm_name, hash_log)
    volatile_count = sum(1 for v in layer1.get("artefacts", {}).values() if v)
    timeline.add("Preservation", f"Layer 1 volatile state collected ({volatile_count}/6 artefacts)")

    tool_check = check_capture_tools(vm_name)
    timeline.add("Preservation", f"Capture tool check: {tool_check.get('sniffer_tools_found', 0)} "
                                 f"known tool(s) present on the file system")

    # the SSH channel closes once the quarantine is applied, so the fallback copy is taken now
    layer2_ssh = collect_layer2_ssh(vm_name, hash_log)
    timeline.add("Preservation", f"Guest-view copy of {len(layer2_ssh.get('artefacts', {}))} disk "
                                 f"artefact(s) taken over SSH before quarantine (fallback source)")
    latencies["volatile_collection_ms"] = _elapsed_ms(t0)

    print(f"\n[10] Isolating suspicious VM: {vm_name} ...")
    isolation = isolate_vm(vm_name)
    if isolation:
        latencies["isolation_ms"] = isolation["latency_ms"]
        timeline.add("Response", f"{vm_name} isolated via {isolation['security_group']} security group")

    print("\n[10b] Preserving PCAP and disk state ...")
    t0 = time.perf_counter()
    case_dir = new_case_dir(vm_name)
    preservation = preserve_disk_and_pcap(vm_name, hash_log, layer1, layer2_ssh, pcap_path, case_dir)
    preservation["binary_hash_check"] = tool_check
    _save_json(alert, os.path.join(case_dir, "alert.json"))
    latencies["preservation_ms"] = _elapsed_ms(t0) + latencies["volatile_collection_ms"]
    timeline.add("Preservation", f"Three-layer evidence collection: "
                                 f"{preservation['total_artefacts']} artefacts, Layer 2 via "
                                 f"{preservation['layer2'].get('method', 'unknown')}, "
                                 f"chain verified={preservation['chain_verified']}, "
                                 f"content verified={preservation['content_verified']}")

    return {
        "vm_name": vm_name,
        "case_id": os.path.basename(case_dir),
        "case_dir": case_dir,
        "hash_log": hash_log,
        "layer1": layer1,
        "layer2_ssh": layer2_ssh,
        "tool_check": tool_check,
        "isolation": isolation,
        "preservation": preservation,
        "latencies": latencies,
    }


# ----------------------------------------------------------------

def incident_response(response, detections, pcap_path, timeline):
    """Log Collection, PCAP Analysis, Correlate, Sniffing Confirmation."""
    print("\n[11] Gathering the working set and analysing forensic evidence ...")
    t0 = time.perf_counter()

    # Log Collection is a gathering step: Layer 1 + Layer 2 records plus infrastructure logs.
    # The guest-view SSH copy contributes what only a live guest can give (the kernel ring
    # buffer) for any file present in both, the read-only image copy takes precedence.
    working_set = {}
    working_set.update(response["layer1"].get("artefacts", {}))
    for name, content in response["layer2_ssh"].get("artefacts", {}).items():
        working_set.setdefault(name, content)
    working_set.update(response["preservation"]["layer2"].get("artefacts", {}))
    infra_logs = collect_infrastructure_logs(response["vm_name"])
    infra_path = os.path.join(response["case_dir"], "infrastructure_logs.json")
    with open(infra_path, "w") as f:
        json.dump(infra_logs, f, indent=2, default=str)
    response["hash_log"].append("infrastructure_logs",
                                json.dumps(infra_logs, sort_keys=True, default=str))

    log_analysis = analyse_logs(working_set, tool_check=response["tool_check"])
    print(f"    Log analysis: {log_analysis['summary']}")
    for name, result in log_analysis["indicator_results"].items():
        print(f"      {name:<16} {'YES' if result['flagged'] else 'no ':<4} {result['detail']}")

    pcap_analysis = analyse_pcap(pcap_path)
    print(f"    PCAP analysis: {len(pcap_analysis['findings'])} finding(s)")

    decision = correlate_and_decide(log_analysis, pcap_analysis, detections)
    correlation_ms = _elapsed_ms(t0)
    print(f"    Verdict: {decision['verdict']} ({decision['confidence']} confidence)")
    timeline.add("Analysis", f"Log collection: {log_analysis['summary']}")
    timeline.add("Analysis", f"PCAP analysis: {len(pcap_analysis['findings'])} finding(s)")
    timeline.add("Analysis", f"Correlation verdict: {decision['verdict']} "
                             f"({decision['confidence']} confidence)")

    return {
        "log_analysis": log_analysis,
        "pcap_analysis": pcap_analysis,
        "infra_logs": infra_logs,
        "decision": decision,
        "correlation_ms": correlation_ms,
        "working_set": working_set,
    }


# ---------------------------------------------------------------- 

def dismiss_alert(alert, response, analysis, detections, threshold, pcap_path, timeline):
    """Verdict Logging first, Restore Security Group only once the record is committed."""
    decision = analysis["decision"]
    print(f"\n[12] Alert dismissed ({decision['verdict']}) -- logging verdict ...")
    indicator_artefacts = ("ps_aux", "ss_tulnp", "ip_link_show", "proc_net_packet", "lsmod")
    log_excerpts = {k: v for k, v in analysis.get("working_set", {}).items()
                    if k in indicator_artefacts and isinstance(v, str)}
    verdict = log_verdict(
        case_id=response["case_id"],
        decision=decision,
        detections=detections,
        threshold=threshold,
        log_analysis=analysis["log_analysis"],
        pcap_analysis=analysis["pcap_analysis"],
        isolation=response["isolation"],
        pcap_path=pcap_path,
        preservation=response["preservation"],
        alert=alert,
        log_excerpts=log_excerpts,
    )
    response["hash_log"].append("verdict_record", json.dumps(verdict["entry"], sort_keys=True,
                                                             default=str))
    # every hashed artefact must exist as a file so the chain can be re-verified later
    response["hash_log"].save_artefacts(response["case_dir"])
    response["hash_log"].write(response["case_dir"])
    print(f"    Verdict logged -> {verdict['file_path']}")
    print(f"    Entry hash: {verdict['entry_hash'][:16]}...")
    timeline.add("Verdict", f"Dismissal recorded ({decision['verdict']}), entry hash "
                            f"{verdict['entry_hash'][:16]}")

    if response["isolation"]:
        print(f"\n[12b] Restoring {response['vm_name']} (removing isolation) ...")
        restore_vm(response["vm_name"], response["isolation"]["security_group"],
                   response["isolation"].get("original_security_groups"))
        timeline.add("Response", f"{response['vm_name']} restored to normal operation")
    return verdict


# ---------------------------------------------------------------- 

def reporting(alert, response, analysis, detections, threshold, pcap_path, output_dir,
              timeline, incident_id=1, stage_latencies=None):
    """Extract timestamps, build the timeline, hash it, write the JSON and PDF report, hash
    the report into the same chain, and send the notification."""
    print("\n[13] Sniffing CONFIRMED -- generating forensic report ...")
    t0 = time.perf_counter()
    hash_log = response["hash_log"]
    case_dir = response["case_dir"]
    preservation = response["preservation"]

    hash_log.append("alert_metadata", json.dumps(alert, sort_keys=True, default=str))
    timeline_events = sorted(timeline.events, key=lambda e: e["timestamp"])
    hash_log.append("timeline", json.dumps(timeline_events, sort_keys=True, default=str))
    print(f"    Timeline: {len(timeline_events)} events")

    report_id = f"FR-{incident_id:04d}"
    timeline.add("Report", f"Forensic report {report_id} generated")
    report = generate_forensic_report(
        detections, pcap_path, incident_id=incident_id, threshold=threshold,
        isolation=response["isolation"], preservation=preservation,
        correlation=analysis["decision"], infra_logs=analysis["infra_logs"],
        shap_features=alert["shap_features"], timeline_events=timeline.events,
        suspect=alert["suspect_vm"],
    )
    if stage_latencies:
        report["stage_latencies_ms"] = stage_latencies

    json_path = _save_json(report, os.path.join(output_dir, "forensic_report.json"))
    print(f"    Forensic report saved -> {json_path}")
    pdf_path = generate_report_pdf(json_path)

    # the report joins the chain after it exists, so the chain file is rewritten last
    for name, path in (("forensic_report_json", json_path), ("forensic_report_pdf", pdf_path)):
        with open(path, "rb") as f:
            hash_log.append(name, f.read())
    preservation["chain_of_custody"] = hash_log.to_json()
    preservation["total_artefacts"] = len(hash_log.entries)
    # the alert, timeline and report entries were appended after the first save, so the
    # artefact files are refreshed before the chain file is rewritten
    hash_log.save_artefacts(case_dir)
    hash_log.write(case_dir)
    shutil.copy2(json_path, case_dir)
    shutil.copy2(pdf_path, case_dir)
    report_ms = _elapsed_ms(t0)

    print("\n[13b] Sending incident notification ...")
    notify_incident(
        {"classification": report["classification"], "severity": report["severity"],
         "suspect_vm": alert["suspect_vm"], "confidence": alert["score"],
         "timestamp": report["generated_at"]},
        report_path=pdf_path,
    )
    return report, json_path, pdf_path, report_ms


def collect_stage_latencies(inference_ms, response, analysis, report_ms=None):
    """Per-stage latencies for Chapter 5, Table tab:pipeline-timings."""
    latencies = {
        "inference_ms": inference_ms,
        "isolation_ms": response["latencies"].get("isolation_ms"),
        "preservation_ms": response["latencies"].get("preservation_ms"),
        "correlation_ms": analysis["correlation_ms"],
    }
    if report_ms is not None:
        latencies["report_ms"] = report_ms
    return latencies
