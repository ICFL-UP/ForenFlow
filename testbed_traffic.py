"""Testbed traffic generators driven from the DHCP namespace (test harness, not part of the model).

Run directly (python3 testbed_traffic.py) to start the guest services the benign generator talks to.
"""
import subprocess
import time

from services.constants import VM_MAP, DHCP_NS, GUEST_USER, GUEST_PASSWORD

ATTACKER_VM = "vm-app-02"


def _spawn(cmd):
    return subprocess.Popen(f"sudo ip netns exec {DHCP_NS} {cmd}", shell=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ssh_guest(vm_ip, cmd):
    """Run a command inside a guest over the DHCP-namespace SSH channel (Chapter 4, 4.3.2)."""
    full = (f"sudo ip netns exec {DHCP_NS} sshpass -p {GUEST_PASSWORD} ssh "
            f"-o StrictHostKeyChecking=no -o ConnectTimeout=8 {GUEST_USER}@{vm_ip} '{cmd}'")
    try:
        return subprocess.run(full, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def start_guest_services():
    """Start the BusyBox web server on TCP/80 in every VM so the benign generator's HTTP probes
    are completed request/response exchanges. Without a listener the probe is a SYN answered by
    RST, which is the closed-port pattern the attack generator produces on purpose. Idempotent,
    and the servers survive until the guest reboots."""
    print("  [Services] Starting BusyBox httpd on port 80 in every VM ...")
    # the check looks at the listening socket, not the process table: CirrOS BusyBox has no
    # pgrep, and a ps/grep check would match this very command line
    for name, v in VM_MAP.items():
        out = _ssh_guest(v["ip"], "mkdir -p /tmp/www && echo \"<html><body>F-RIaaS testbed</body></html>\" "
                                  "> /tmp/www/index.html && chmod -R a+rX /tmp/www && "
                                  "(netstat -tln 2>/dev/null | grep -q \":80 \" || sudo httpd -p 80 -h /tmp/www); "
                                  "sleep 1; netstat -tln 2>/dev/null | grep -c \":80 \"")
        print(f"    {name} ({v['ip']}): {'httpd listening on :80' if out.strip() not in ('', '0') else 'NO listener on :80'}")


def generate_normal_traffic(duration_sec=30, rounds=1):
    """Pings between every VM pair plus a few HTTP and DNS probes: the benign baseline."""
    print("  [Traffic] Generating NORMAL traffic patterns ...")
    procs = []
    ips = [v["ip"] for v in VM_MAP.values()]
    ping_count = max(duration_sec // rounds, 1)

    for _ in range(rounds):
        for src_ip in ips:
            for dst_ip in ips:
                if src_ip != dst_ip:
                    procs.append(_spawn(f"ping -c {ping_count} -i 1 {dst_ip}"))

        # a complete HTTP/1.0 request so the guest web server (start_guest_services) answers it
        for dst_ip in ips[:2]:
            for _ in range(5):
                procs.append(_spawn(
                    f"bash -c 'printf \"GET / HTTP/1.0\\r\\n\\r\\n\" | nc -w 1 {dst_ip} 80 "
                    f"2>/dev/null; true'"))
            time.sleep(2)

        for dst_ip in ips[2:4]:
            for _ in range(10):
                procs.append(_spawn(
                    f"bash -c 'echo query | nc -u -w 1 {dst_ip} 53 2>/dev/null; true'"))
            time.sleep(1)

    return procs


def generate_attack_traffic(rounds=1, ports=(22, 80, 443, 3306, 8080, 5000),
                            ping_count=50, connections_per_port=20, exfil_kb=100):
    """Reconnaissance ping sweep, SYN bursts across service ports, then bulk exfiltration."""
    print("  [Traffic] Generating ATTACK traffic patterns ...")
    procs = []
    victim_ips = [v["ip"] for name, v in VM_MAP.items() if name != ATTACKER_VM]

    for round_num in range(rounds):
        if rounds > 1:
            print(f"    Attack round {round_num + 1}/{rounds} ...")

        print("    Phase 1: Reconnaissance scan ...")
        for victim in victim_ips:
            procs.append(_spawn(f"ping -c {ping_count} -i 0.05 -s 64 {victim}"))
        time.sleep(5)

        print("    Phase 2: SYN bursts across service ports ...")
        for victim in victim_ips:
            for port in ports:
                procs.append(_spawn(
                    f"bash -c 'for i in $(seq 1 {connections_per_port}); do "
                    f"echo SNIFF_DATA_$i | nc -w 1 {victim} {port} 2>/dev/null; done'"))
        time.sleep(8)

        print("    Phase 3: Data exfiltration ...")
        for victim in victim_ips:
            procs.append(_spawn(
                f"bash -c 'dd if=/dev/urandom bs=1024 count={exfil_kb} 2>/dev/null | "
                f"nc -w 2 {victim} 9999 2>/dev/null; true'"))

    return procs


def generate_benign_icmp_load(duration_sec, pings_per_second, targets=None):
    """Steady ICMP echo at a fixed rate per target, used for the false positive load curve."""
    targets = targets or [v["ip"] for name, v in VM_MAP.items() if name != ATTACKER_VM]
    if pings_per_second <= 0:
        return []
    interval = 1.0 / pings_per_second
    return [_spawn(f"timeout {duration_sec} ping -i {interval:.3f} -q {ip}")
            for ip in targets]


def stop_all(procs):
    for proc in procs:
        try:
            proc.kill()
        except OSError:
            pass


if __name__ == "__main__":
    start_guest_services()
