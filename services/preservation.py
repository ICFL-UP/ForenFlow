"""Three-layer evidence preservation and the chained SHA-256 hash log)."""
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from services.constants import (VM_MAP, DHCP_NS, GUEST_USER, GUEST_PASSWORD,
                                SNIFFER_TOOLS, SNIFFER_HASH_BASELINE, EVIDENCE_DIR)


@dataclass
class HashLogEntry:
    index: int
    artefact: str
    sha256: str
    timestamp: str
    chained_hash: str


class ChainedHashLog:
    """Each entry folds in the chained hash of the previous entry, so altering or removing any
    past artefact breaks every hash after it."""

    def __init__(self):
        self.entries = []
        self._raw_content = {}

    @staticmethod
    def _chain(index, artefact, sha256, timestamp, prev):
        return hashlib.sha256(f"{index}|{artefact}|{sha256}|{timestamp}|{prev}".encode()).hexdigest()

    def append(self, artefact_name, content):
        if artefact_name in self._raw_content:
            raise ValueError(f"artefact '{artefact_name}' is already in the chain")
        raw = content if isinstance(content, bytes) else content.encode()
        self._raw_content[artefact_name] = raw
        prev = self.entries[-1].chained_hash if self.entries else "GENESIS"
        entry = HashLogEntry(
            index=len(self.entries) + 1,
            artefact=artefact_name,
            sha256=hashlib.sha256(raw).hexdigest(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            chained_hash="",
        )
        entry.chained_hash = self._chain(entry.index, entry.artefact, entry.sha256,
                                         entry.timestamp, prev)
        self.entries.append(entry)
        return entry

    def verify(self):
        """Rebuild the chain from GENESIS, returns (ok, index of the first broken entry or -1)."""
        prev = "GENESIS"
        for i, entry in enumerate(self.entries):
            if self._chain(entry.index, entry.artefact, entry.sha256,
                           entry.timestamp, prev) != entry.chained_hash:
                return False, i
            prev = entry.chained_hash
        return True, -1

    @staticmethod
    def _artefact_path(case_dir, artefact_name):
        return os.path.join(case_dir, "artefacts",
                            artefact_name.replace("/", "_").replace("\\", "_"))

    def save_artefacts(self, case_dir):
        os.makedirs(os.path.join(case_dir, "artefacts"), exist_ok=True)
        for name, raw in self._raw_content.items():
            with open(self._artefact_path(case_dir, name), "wb") as f:
                f.write(raw)

    def verify_content(self, case_dir):
        """Re-hash the saved artefact files against the log this is the check an investigator
        runs later, independently of the in-memory chain."""
        for i, entry in enumerate(self.entries):
            path = self._artefact_path(case_dir, entry.artefact)
            if not os.path.exists(path):
                return False, i
            with open(path, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != entry.sha256:
                    return False, i
        return True, -1

    def to_json(self):
        return [asdict(e) for e in self.entries]

    def write(self, case_dir):
        path = os.path.join(case_dir, "chain_of_custody.json")
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        return path


def _run_in_guest(cmd, vm_ip):
    """Run a command inside the guest over SSH through the DHCP namespace."""
    full = (f"sudo ip netns exec {DHCP_NS} sshpass -p {GUEST_PASSWORD} "
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "
            f"{GUEST_USER}@{vm_ip} '{cmd}'")
    try:
        return subprocess.run(full, shell=True, capture_output=True,
                              text=True, timeout=15).stdout
    except subprocess.TimeoutExpired:
        return ""


def _vm_ip(vm_name):
    return VM_MAP.get(vm_name, {}).get("ip", "")


# The six volatile artefacts of Layer 1. Keys are the artefact names that
# appear in the chain of custody, so they are kept stable across runs.
LAYER1_COMMANDS = {
    "ps_aux": "ps aux",
    "ss_tulnp": "ss -tulnp 2>/dev/null || netstat -tlnp 2>/dev/null",
    "ip_link_show": "ip link show",
    "proc_net_packet": "cat /proc/net/packet 2>/dev/null",
    "lsmod": "lsmod 2>/dev/null || cat /proc/modules",
    # descriptors of every process, not of the ls process itself
    "open_fds": "sudo ls -la /proc/[0-9]*/fd 2>/dev/null",
}

# Layer 2 via SSH: CirrOS keeps its system log in /var/log/messages and has no separate auth
# log, so the kernel ring buffer stands in for the promiscuous-mode message source.
LAYER2_SSH_COMMANDS = {
    "messages": "cat /var/log/messages 2>/dev/null",
    "dmesg": "sudo dmesg 2>/dev/null || dmesg 2>/dev/null || cat /var/log/dmesg 2>/dev/null",
    "ash_history": "cat /root/.ash_history 2>/dev/null || cat ~/.ash_history 2>/dev/null || "
                   "cat /home/cirros/.ash_history 2>/dev/null",
}

# Layer 2 via libguestfs: the same artefact classes read from the read-only disk image
LAYER2_GUEST_FILES = {
    "messages": "/var/log/messages",
    "acpid_log": "/var/log/acpid.log",
    "dhcpcd_conf": "/etc/dhcpcd.conf",
    "ash_history": "/root/.ash_history",
}

# One guestfish session copies every file, so the appliance boots once per collection. Without
# KVM (a compute host that is itself a virtual machine) the appliance runs under software
# emulation and a boot takes minutes rather than seconds, hence the generous limit.
GUESTFISH_TIMEOUT_SECONDS = 600


def _guestfish_download_all(img_path, out_dir):
    """Read-only mount of the image, then one download per artefact. The leading '-' tells
    guestfish to carry on when a file does not exist in the guest (no history yet, for example)."""
    script = "".join(f"-download {guest_path} {os.path.join(out_dir, name)}\n"
                     for name, guest_path in LAYER2_GUEST_FILES.items())
    subprocess.run(["guestfish", "--ro", "-a", img_path, "-i"], input=script,
                   capture_output=True, text=True, timeout=GUESTFISH_TIMEOUT_SECONDS, check=True)


def collect_layer1(vm_name, hash_log):
    """Layer 1, Collect Volatile State, run BEFORE quarantine while the guest is reachable."""
    print("    [Layer 1] Collecting volatile state ...")
    vm_ip = _vm_ip(vm_name)
    if not vm_ip:
        return {"status": "skipped", "reason": "no IP for VM", "layer": 1, "artefacts": {}}

    t0 = time.perf_counter()
    artefacts = {}
    for name, cmd in LAYER1_COMMANDS.items():
        artefacts[name] = _run_in_guest(cmd, vm_ip)
        hash_log.append(name, artefacts[name])

    duration_ms = (time.perf_counter() - t0) * 1000
    collected = sum(1 for v in artefacts.values() if v)
    print(f"      Collected {collected}/{len(LAYER1_COMMANDS)} volatile artefacts "
          f"in {duration_ms:.1f} ms")
    return {"artefacts": artefacts, "duration_ms": round(duration_ms, 2), "layer": 1}


def collect_layer2_ssh(vm_name, hash_log):
    """Guest-view copy of the Layer 2 files over SSH, taken BEFORE quarantine because the
    channel closes afterwards. It is the fallback source when the read-only image mount fails,
    and is chained under its own names (disk_ssh_*) so it never collides with the mount copy.
    """

    print("    [Layer 2 SSH] Taking guest-view copy of disk artefacts (fallback source) ...")
    vm_ip = _vm_ip(vm_name)
    if not vm_ip:
        return {"status": "skipped", "reason": "no IP for VM", "layer": 2,
                "method": "ssh", "artefacts": {}}

    t0 = time.perf_counter()
    artefacts = {}
    for name, cmd in LAYER2_SSH_COMMANDS.items():
        artefacts[name] = _run_in_guest(cmd, vm_ip).strip()
        hash_log.append(f"disk_ssh_{name}", artefacts[name])

    duration_ms = (time.perf_counter() - t0) * 1000
    non_empty = sum(1 for v in artefacts.values() if v)
    print(f"      Copied {len(artefacts)} disk artefact(s) "
          f"({non_empty} non-empty) over SSH in {duration_ms:.1f} ms")
    return {"artefacts": artefacts, "duration_ms": round(duration_ms, 2),
            "layer": 2, "method": "ssh"}


def _create_server_snapshot(conn, vm_name, prefix):
    server = conn.compute.find_server(vm_name)
    snapshot_name = f"{prefix}-{vm_name}-{int(time.time())}"
    image = conn.compute.create_server_image(server, snapshot_name)
    return snapshot_name, (image if isinstance(image, str) else image.id)


def collect_layer2_libguestfs(vm_name, hash_log):
    """Layer 2: read the files from a read-only image of the disk,
    after quarantin."""

    print("    [Layer 2] Collecting disk artefacts via libguestfs ...")
    from services.openstack_service import _get_connection

    t0 = time.perf_counter()
    artefacts = {}
    conn = None
    snapshot_name = None
    img_path = None
    try:
        conn = _get_connection()
        snapshot_name, snap_id = _create_server_snapshot(conn, vm_name, "friaas-evidence")
        for _ in range(30):
            if conn.image.get_image(snap_id).status == "active":
                break
            time.sleep(2)

        img_dir = "/tmp/friaas_evidence"
        out_dir = os.path.join(img_dir, f"{snapshot_name}-files")
        os.makedirs(out_dir, exist_ok=True)
        img_path = os.path.join(img_dir, f"{snapshot_name}.qcow2")
        with open(img_path, "wb") as f:
            for chunk in conn.image.download_image(snap_id):
                f.write(chunk)

        _guestfish_download_all(img_path, out_dir)
        # a file that exists but is empty is still evidence
        for name in LAYER2_GUEST_FILES:
            path = os.path.join(out_dir, name)
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                content = f.read().decode(errors="replace")
            os.remove(path)
            artefacts[name] = content
            hash_log.append(f"disk_{name}", content)
        os.rmdir(out_dir)
    except Exception as exc:
        print(f"      Layer 2 libguestfs collection error: {exc}")
    finally:
        # the temporary image is not evidence (Layer 3 keeps the real snapshot), so remove it
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        if conn and snapshot_name:
            try:
                image = conn.image.find_image(snapshot_name)
                if image:
                    conn.image.delete_image(image.id)
            except Exception as exc:
                print(f"      Could not delete temporary image {snapshot_name}: {exc}")

    duration_ms = (time.perf_counter() - t0) * 1000
    print(f"      Collected {len(artefacts)} disk artefact(s) via libguestfs in {duration_ms:.1f} ms")
    return {"artefacts": artefacts, "duration_ms": round(duration_ms, 2),
            "layer": 2, "method": "libguestfs"}


def collect_layer3(vm_name, hash_log):
    """Layer 3, copy-on-write snapshot. The snapshot stays in the image store, so the chain
    records its identifier and creation time rather than a content hash."""

    print("    [Layer 3] Creating CoW snapshot ...")
    from services.openstack_service import _get_connection

    t0 = time.perf_counter()
    try:
        snapshot_name, snap_id = _create_server_snapshot(_get_connection(), vm_name, "friaas-cow")
        hash_log.append("cow_snapshot", f"snapshot:{snapshot_name}:id:{snap_id}")
        duration_ms = (time.perf_counter() - t0) * 1000
        print(f"      CoW snapshot created: {snapshot_name} in {duration_ms:.1f} ms")
        return {"snapshot_name": snapshot_name, "snapshot_id": snap_id,
                "duration_ms": round(duration_ms, 2), "layer": 3}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc),
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2), "layer": 3}


def check_capture_tools(vm_name):
    """Tool presence and hash check (indicator 5): which known capture tools are installed on
    the guest file system, with SHA-256 compared to the baseline where one exists. Runs during
    Layer 1 collection because it needs the guest channel."""

    print(f"  [HashCheck] Checking capture tools on {vm_name} ...")
    vm_ip = _vm_ip(vm_name)
    if not vm_ip:
        return {"status": "skipped", "reason": "VM IP not found",
                "binaries_checked": [], "sniffer_tools_found": 0, "findings": []}

    binaries_checked = []
    findings = []
    for tool in SNIFFER_TOOLS:
        bin_path = _run_in_guest(f"which {tool} 2>/dev/null", vm_ip).strip()
        if not bin_path:
            binaries_checked.append({"binary": tool, "present": False})
            continue

        hash_out = _run_in_guest(f"sha256sum {bin_path} 2>/dev/null", vm_ip).split()
        bin_hash = hash_out[0] if hash_out else "unreadable"
        expected = SNIFFER_HASH_BASELINE.get(tool)
        entry = {"binary": tool, "path": bin_path, "sha256": bin_hash, "present": True,
                 "baseline_match": (bin_hash == expected) if expected else None}
        binaries_checked.append(entry)
        findings.append(f"Sniffer tool found: {tool} at {bin_path}")
        if expected and bin_hash != expected:
            findings.append(f"{tool} at {bin_path}: hash mismatch against baseline")

    return {
        "status": "completed",
        "binaries_checked": binaries_checked,
        "sniffer_tools_found": sum(1 for b in binaries_checked if b["present"]),
        "findings": findings,
    }


def preserve_disk_and_pcap(vm_name, hash_log, layer1, layer2_ssh, pcap_path, case_dir):
    """Preserve PCAP and Disk State , run AFTER quarantine: try the read-only
    image mount for Layer 2, fall back to the SSH copy taken before quarantine, freeze the
    volume, copy the capture segment into the repository, then save and verify the chain."""
    
    print(f"  [Preserve] Preserving disk state and PCAP for {vm_name} ...")

    layer2 = collect_layer2_libguestfs(vm_name, hash_log)
    if not layer2["artefacts"]:
        print("      Read-only image mount yielded nothing, using the SSH guest-view copy for Layer 2")
        layer2 = layer2_ssh

    layer3 = collect_layer3(vm_name, hash_log)

    if pcap_path and os.path.exists(pcap_path):
        with open(pcap_path, "rb") as f:
            hash_log.append("pcap_capture", f.read())

    os.makedirs(case_dir, exist_ok=True)
    hash_log.save_artefacts(case_dir)
    chain_ok, _ = hash_log.verify()
    content_ok, _ = hash_log.verify_content(case_dir)
    hash_log.write(case_dir)

    total_ms = sum(layer.get("duration_ms", 0) for layer in (layer1, layer2, layer3))
    print(f"    Total preservation: {total_ms:.1f} ms, {len(hash_log.entries)} artefacts, "
          f"chain verified={chain_ok}, content verified={content_ok}")

    return {
        "layer1": layer1,
        "layer2": layer2,
        "layer3": layer3,
        "chain_of_custody": hash_log.to_json(),
        "chain_verified": chain_ok,
        "content_verified": content_ok,
        "total_artefacts": len(hash_log.entries),
        "total_duration_ms": round(total_ms, 2),
        "evidence_dir": case_dir,
    }


def new_case_dir(vm_name):
    return os.path.join(EVIDENCE_DIR, f"case-{vm_name}-{int(time.time())}")
