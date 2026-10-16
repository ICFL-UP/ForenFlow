"""Incident Response analysis: five host-side indicators, PCAP inspection, three-source verdict."""
import os
import re

from services.constants import SNIFFER_TOOLS

# PCAP analysis thresholds: the model names the patterns, the prototype sets
# the counts that qualify as "high" for this testbed
PORT_DIVERSITY_THRESHOLD = 20
ICMP_COUNT_THRESHOLD = 100
SYN_ONLY_THRESHOLD = 50

# Layer 2 artefact keys that hold the supporting log evidence for indicator 4
LOG_ARTEFACTS = ("messages", "dmesg", "ash_history", "auth_log", "acpid_log")

_TOOL_ALTERNATION = "|".join(SNIFFER_TOOLS)

# Indicator 4: each pattern is one kind of supporting evidence
LOG_MARKERS = {
    "capture written to file": rf"(?:{_TOOL_ALTERNATION})\b[^\n]*\s-w\s|(?:{_TOOL_ALTERNATION})\b[^\n]*>\s*\S+\.pcap",
    "capture streamed to host": rf"(?:{_TOOL_ALTERNATION})\b[^\n]*\|\s*(?:nc|ncat|ssh|curl)\b",
    "capture tool invoked": rf"\b(?:{_TOOL_ALTERNATION})\b",
    "promiscuous mode command": r"ip\s+link\s+set\s+\S+\s+promisc\s+on|ifconfig\s+\S+\s+promisc",
    "kernel promiscuous message": r"entered promiscuous mode",
    "scheduled capture": rf"(?:cron|crontab|systemd-run|\.timer)[^\n]*(?:{_TOOL_ALTERNATION})",
    "privilege escalation": rf"\b(?:sudo|su\s+-c?)\b[^\n]*\b(?:{_TOOL_ALTERNATION})\b",
    "unauthorised login": r"Failed password|authentication failure|Invalid user",
}


def _check_promiscuous(evidence):
    """Indicator 1: PROMISC flag on an interface (Layer 1), or the kernel message where the
    guest exposes the ring buffer."""
    if re.search(r"\bPROMISC\b", evidence.get("ip_link_show", "")):
        return True, "PROMISC flag set in ip link show"
    kernel = evidence.get("dmesg", "") + evidence.get("messages", "")
    if re.search(r"entered promiscuous mode", kernel):
        return True, "kernel reported 'entered promiscuous mode'"
    return False, "No PROMISC flag observed"


def _check_af_packet(evidence):
    """Indicator 2: system-wide AF_PACKET sockets from /proc/net/packet (Layer 1)."""
    lines = [ln for ln in evidence.get("proc_net_packet", "").splitlines() if ln.strip()]
    if not lines:
        return False, "proc/net/packet not available"
    sockets = lines[1:]  # first line is the column header
    if sockets:
        return True, f"{len(sockets)} active AF_PACKET socket(s)"
    return False, "Not detected"


def _check_sniffer_process(evidence):
    """Indicator 3: name-based match of known capture tools in the process listing (Layer 1)."""
    ps_out = evidence.get("ps_aux", "")
    found = [tool for tool in SNIFFER_TOOLS if re.search(rf"\b{tool}\b", ps_out)]
    if found:
        return True, f"sniffer process(es): {', '.join(found)}"
    return False, "No sniffer processes detected"


def _check_log_markers(evidence):
    """Indicator 4: supporting evidence in the shell history, system and auth logs (Layer 2)."""
    blob = "\n".join(evidence.get(key, "") for key in LOG_ARTEFACTS)
    hits = {name: len(re.findall(pattern, blob, re.IGNORECASE))
            for name, pattern in LOG_MARKERS.items()}
    hits = {name: count for name, count in hits.items() if count}
    if hits:
        detail = ", ".join(f"{name} x{count}" for name, count in hits.items())
        return True, f"{sum(hits.values())} log marker(s): {detail}"
    return False, "No log indicators found"


def _check_tool_presence(tool_check):
    """Indicator 5: known capture tools installed on the file system, hash-verified where a
    baseline exists (collected by preservation.check_capture_tools)."""
    if not tool_check or tool_check.get("status") != "completed":
        return False, "tool presence check not performed"
    present = [b for b in tool_check["binaries_checked"] if b.get("present")]
    if not present:
        return False, "no known capture tools installed"
    detail = ", ".join(f"{b['binary']} installed at {b['path']}" for b in present)
    mismatched = [b["binary"] for b in present if b.get("baseline_match") is False]
    if mismatched:
        detail += f"; hash mismatch: {', '.join(mismatched)}"
    return True, detail


def analyse_logs(evidence, tool_check=None):
    """Log Collection: evaluate the five indicators against the working set
    (Layer 1 + Layer 2 artefacts). Any flagged indicator makes the host-side source suspicious."""
    checks = {
        "promiscuous": _check_promiscuous(evidence),
        "af_packet": _check_af_packet(evidence),
        "sniffer_process": _check_sniffer_process(evidence),
        "log_markers": _check_log_markers(evidence),
        "hash_baseline": _check_tool_presence(tool_check),
    }
    indicator_results = {name: {"flagged": flagged, "detail": detail}
                         for name, (flagged, detail) in checks.items()}
    flagged = sum(1 for r in indicator_results.values() if r["flagged"])
    return {
        "suspicious": flagged > 0,
        "indicator_results": indicator_results,
        "indicators_flagged": flagged,
        "summary": (f"{flagged}/5 forensic indicator(s) flagged"
                    if flagged else "No indicators found"),
    }


def _pcap_findings(dst_ports, icmp_count, syn_count):
    findings = []
    if len(dst_ports) > PORT_DIVERSITY_THRESHOLD:
        findings.append(f"High destination port diversity: {len(dst_ports)} unique ports "
                        f"(scanning indicator)")
    if icmp_count > ICMP_COUNT_THRESHOLD:
        findings.append(f"ICMP flood detected: {icmp_count} ICMP packets "
                        f"(reconnaissance indicator)")
    if syn_count > SYN_ONLY_THRESHOLD:
        findings.append(f"SYN flood detected: {syn_count} SYN-only packets")
    return findings


def analyse_pcap_scapy(pcap_path):
    from scapy.all import rdpcap, IP, TCP, UDP

    dst_ports = set()
    icmp_count = syn_count = 0
    packets = rdpcap(pcap_path)
    for pkt in packets:
        if not pkt.haslayer(IP):
            continue
        if pkt[IP].proto == 1:
            icmp_count += 1
        if pkt.haslayer(TCP):
            dst_ports.add(pkt[TCP].dport)
            if pkt[TCP].flags & 0x02 and not pkt[TCP].flags & 0x10:
                syn_count += 1
        elif pkt.haslayer(UDP):
            dst_ports.add(pkt[UDP].dport)

    findings = _pcap_findings(dst_ports, icmp_count, syn_count)
    return {"suspicious": bool(findings), "findings": findings,
            "packets_analysed": len(packets), "tool": "scapy"}


def analyse_pcap_pyshark(pcap_path):
    import pyshark

    dst_ports = set()
    icmp_count = syn_count = total = 0
    cap = pyshark.FileCapture(pcap_path, keep_packets=False)
    for pkt in cap:
        total += 1
        if hasattr(pkt, "icmp"):
            icmp_count += 1
        if hasattr(pkt, "tcp"):
            dst_ports.add(int(pkt.tcp.dstport))
            if pkt.tcp.flags_syn == "1" and pkt.tcp.flags_ack == "0":
                syn_count += 1
        elif hasattr(pkt, "udp"):
            dst_ports.add(int(pkt.udp.dstport))
    cap.close()

    findings = _pcap_findings(dst_ports, icmp_count, syn_count)
    return {"suspicious": bool(findings), "findings": findings,
            "packets_analysed": total, "tool": "pyshark"}


def analyse_pcap(pcap_path):
    """PCAP Analysis: Scapy first, PyShark when Scapy is unavailable."""
    if not pcap_path or not os.path.exists(pcap_path):
        return {"suspicious": False, "findings": ["No PCAP available"], "packets_analysed": 0}
    for analyser in (analyse_pcap_scapy, analyse_pcap_pyshark):
        try:
            return analyser(pcap_path)
        except ImportError:
            continue
    return {"suspicious": False, "packets_analysed": 0,
            "findings": ["Neither Scapy nor PyShark available for PCAP analysis"]}


def correlate_and_decide(log_analysis, pcap_analysis, detections):
    """Correlate and confirm: majority vote over the host-side,
    packet-side and detection sources. CONFIRMED needs two of three."""
    ml_flagged = any(d.get("detected") for d in detections)
    logs_suspicious = bool(log_analysis.get("suspicious"))
    pcap_suspicious = bool(pcap_analysis.get("suspicious"))
    sources_flagged = sum([ml_flagged, logs_suspicious, pcap_suspicious])

    if sources_flagged >= 2:
        verdict, confidence = "CONFIRMED", "HIGH"
    elif sources_flagged == 1:
        verdict, confidence = "UNCERTAIN", "MEDIUM"
    else:
        verdict, confidence = "BENIGN", "LOW"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "restore_vm": verdict != "CONFIRMED",
        "ml_detected": ml_flagged,
        "logs_suspicious": logs_suspicious,
        "pcap_suspicious": pcap_suspicious,
        "sources_flagged": sources_flagged,
        "summary": f"Verdict: {verdict} ({sources_flagged}/3 sources flagged)",
    }
