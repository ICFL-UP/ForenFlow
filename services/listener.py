"""Listener service: capture a segment from the mirror port and extract per-flow features."""
import subprocess
from collections import defaultdict

import numpy as np

import config
from services.constants import IP_TO_VM, MIRROR_IFACE


def capture_traffic(output_pcap, duration_sec=60, max_packets=50000):
    """Start tcpdump on the mirror interface for one capture segment."""
    print(f"  [Capture] Starting tcpdump on {MIRROR_IFACE} for {duration_sec}s ...")
    # +5s so tcpdump can flush its buffer before timeout kills it
    cmd = (f"sudo timeout {duration_sec + 5} tcpdump -i {MIRROR_IFACE} "
           f"-w {output_pcap} -c {max_packets} 2>&1")
    return subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _flow_record(src_ip, dst_ip, proto, src_port, dst_port, num_packets, duration,
                 mean_packet_size, mean_gap, active_time, idle_time, flag_count,
                 packet_size_stddev, bytes_sent, bytes_received, first_seen, flow_key):
    """Single place that lays out the 18-feature vector so both extractors agree."""
    duration = max(duration, 0.001)
    bytes_received = max(bytes_received, 1)
    return {
        "PacketSize": float(mean_packet_size),
        "ResponseTime": float(mean_gap),
        "Protocol": proto,
        "SrcPort": src_port,
        "DstPort": dst_port,
        "PacketRate": num_packets / duration,
        "FlowDuration": duration,
        "NumPackets": num_packets,
        "PayloadSize": float(mean_packet_size) * 0.8,
        "FlagCount": flag_count,
        "Entropy": float(packet_size_stddev),
        "BytesSent": bytes_sent,
        "BytesReceived": bytes_received,
        "FlowRate": bytes_sent / duration,
        "ActiveTime": active_time,
        "IdleTime": idle_time,
        "ByteRatio": bytes_sent / bytes_received,
        "TimeGap": float(mean_gap),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_vm": IP_TO_VM.get(src_ip, "external"),
        "dst_vm": IP_TO_VM.get(dst_ip, "external"),
        "timestamp": first_seen,
        "flow_key": flow_key,
    }


def _extract_nfstream(pcap_path):
    """Primary extractor: NFStream bidirectional flows with the training flow window."""
    from nfstream import NFStreamer

    streamer = NFStreamer(
        source=pcap_path,
        statistical_analysis=True,
        active_timeout=config.FLOW_WINDOW_SECONDS,
        idle_timeout=config.FLOW_WINDOW_SECONDS,
    )

    records = []
    for flow in streamer:
        num_packets = flow.bidirectional_packets
        if num_packets < config.MIN_PACKETS_PER_FLOW:
            continue

        duration = max(flow.bidirectional_duration_ms / 1000.0, 0.001)
        mean_gap = (flow.bidirectional_mean_piat_ms / 1000.0
                    if flow.bidirectional_mean_piat_ms > 0
                    else duration / max(num_packets - 1, 1))
        active_time = min(mean_gap * (num_packets - 1), duration)

        flag_count = 0
        if flow.protocol == 6:
            for attr in ("bidirectional_syn_packets", "bidirectional_fin_packets",
                         "bidirectional_rst_packets", "bidirectional_psh_packets",
                         "bidirectional_ack_packets"):
                flag_count += getattr(flow, attr, 0)

        records.append(_flow_record(
            src_ip=flow.src_ip, dst_ip=flow.dst_ip, proto=flow.protocol,
            src_port=flow.src_port, dst_port=flow.dst_port,
            num_packets=num_packets, duration=duration,
            mean_packet_size=flow.bidirectional_mean_ps, mean_gap=mean_gap,
            active_time=active_time, idle_time=max(duration - active_time, 0.0),
            flag_count=flag_count,
            packet_size_stddev=getattr(flow, "bidirectional_stddev_ps", 0.0) or 0.0,
            bytes_sent=flow.src2dst_bytes, bytes_received=flow.dst2src_bytes,
            first_seen=flow.bidirectional_first_seen_ms / 1000.0,
            flow_key=f"{flow.src_ip}->{flow.dst_ip}:{flow.protocol}@{flow.id}",
        ))

    print(f"    NFStream extracted {len(records)} flow records")
    return records


TCP_FLAG_MARKERS = ("[S]", "[S.]", "[P.]", "[F.]", "[R.]", "[.]")


def _parse_tcpdump_line(line):
    """Turn one 'tcpdump -nn -tt' line into a packet dict, or None if it is not a packet."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        ts = float(parts[0])
    except ValueError:
        return None

    proto = 1 if "ICMP" in line else 17 if "UDP" in line else 6

    try:
        arrow = parts.index(">")
    except ValueError:
        return None
    src_ip, src_port = _split_endpoint(parts[arrow - 1])
    dst_ip, dst_port = _split_endpoint(parts[arrow + 1])
    if not src_ip or not dst_ip:
        return None

    size = 64
    if "length" in parts:
        try:
            size = int(parts[parts.index("length") + 1].rstrip(",):"))
        except (IndexError, ValueError):
            pass

    return {
        "ts": ts, "src_ip": src_ip, "dst_ip": dst_ip,
        "src_port": src_port, "dst_port": dst_port,
        "proto": proto, "size": max(size, 64),
        "flags": sum(1 for marker in TCP_FLAG_MARKERS if marker in line),
    }


def _split_endpoint(token):
    """'10.0.0.5.443' -> ('10.0.0.5', 443). An ICMP endpoint is a bare IPv4 address (four
    octets), so only a fifth dotted part is read as a port."""
    parts = token.rstrip(",:").split(".")
    if len(parts) == 5 and parts[4].isdigit():
        return ".".join(parts[:4]), int(parts[4])
    return ".".join(parts), 0


def _extract_tcpdump(pcap_path, max_lines=50000, min_packets=None):
    """Fallback extractor: parse tcpdump text and group packets into 5-second sub-flows."""
    if min_packets is None:
        min_packets = config.MIN_PACKETS_PER_FLOW
    cmd = f"sudo tcpdump -r {pcap_path} -nn -tt 2>/dev/null | head -{max_lines}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = [ln for ln in result.stdout.strip().split("\n") if ln]
    if not lines:
        print("    WARNING: No packets captured!")
        return []

    print(f"    Parsing {len(lines)} packet lines via tcpdump fallback ...")
    packets = [p for p in (_parse_tcpdump_line(ln) for ln in lines) if p]
    print(f"    Parsed {len(packets)} packets")

    window_sec = float(config.FLOW_WINDOW_SECONDS)
    flow_windows = defaultdict(list)
    for pkt in packets:
        key = (pkt["src_ip"], pkt["dst_ip"], pkt["proto"])
        flow_windows[(key, int(pkt["ts"] / window_sec))].append(pkt)

    records = []
    for (key, window_idx), pkts in flow_windows.items():
        if len(pkts) < min_packets:
            continue
        src_ip, dst_ip, proto = key
        timestamps = [p["ts"] for p in pkts]
        sizes = [p["size"] for p in pkts]
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
        reverse = flow_windows.get(((dst_ip, src_ip, proto), window_idx), [])

        src_ports = [p["src_port"] for p in pkts]
        dst_ports = [p["dst_port"] for p in pkts]
        records.append(_flow_record(
            src_ip=src_ip, dst_ip=dst_ip, proto=proto,
            src_port=max(set(src_ports), key=src_ports.count),
            dst_port=max(set(dst_ports), key=dst_ports.count),
            num_packets=len(pkts), duration=timestamps[-1] - timestamps[0],
            mean_packet_size=np.mean(sizes),
            mean_gap=np.mean(gaps) if gaps else 0.0,
            # one second separates active transmission from idle waiting
            active_time=sum(g for g in gaps if g < 1.0),
            idle_time=sum(g for g in gaps if g >= 1.0),
            flag_count=sum(p["flags"] for p in pkts),
            packet_size_stddev=np.std(sizes) if len(sizes) > 1 else 0.0,
            bytes_sent=sum(sizes),
            bytes_received=sum(p["size"] for p in reverse),
            first_seen=timestamps[0],
            flow_key=f"{src_ip}->{dst_ip}:{proto}@w{window_idx}",
        ))

    print(f"    Extracted {len(records)} flow records")
    return records


def extract_features_from_pcap(pcap_path):
    """Flow Feature Extraction: NFStream first, tcpdump text parsing otherwise."""
    print("  [Extract] Extracting flow features from PCAP ...")
    try:
        import nfstream  # noqa: F401
    except ImportError:
        print("    NFStream not available, falling back to tcpdump parsing")
        return _extract_tcpdump(pcap_path)
    return _extract_nfstream(pcap_path)
