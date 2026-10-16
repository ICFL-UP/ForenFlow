#!/usr/bin/env python3
"""Discover the testbed VMs, ports and DHCP namespace and rewrite the deployment values in
services/constants.py. Everything else in that file is copied through unchanged."""
import os
import re
import subprocess

import openstack

os.environ.update({
    "OS_AUTH_URL": "http://192.168.100.1/identity",
    "OS_PROJECT_NAME": "admin",
    "OS_USERNAME": "admin",
    "OS_PASSWORD": "secret",
    "OS_USER_DOMAIN_NAME": "Default",
    "OS_PROJECT_DOMAIN_NAME": "Default",
})

VM_NAMES = ["vm-web-01", "vm-app-02", "vm-db-03", "vm-monitor-04"]
ROLES = {"vm-web-01": "web", "vm-app-02": "app", "vm-db-03": "db", "vm-monitor-04": "monitor"}
CONSTANTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "services", "constants.py")


def discover():
    conn = openstack.connect()
    vm_map, known_ids = {}, {}
    for name in VM_NAMES:
        server = conn.compute.find_server(name)
        if not server:
            continue
        ports = list(conn.network.ports(device_id=server.id))
        ip = next((ips[0]["addr"] for ips in (server.addresses or {}).values() if ips), "")
        vm_map[name] = {"ip": ip, "mac": ports[0].mac_address if ports else "", "role": ROLES[name]}
        known_ids[name] = {"port_id": ports[0].id if ports else "unknown", "server_id": server.id}
        print(f"  {name}: ip={ip}, mac={vm_map[name]['mac']}")

    namespaces = subprocess.run("sudo ip netns list", shell=True,
                                capture_output=True, text=True).stdout.split()
    dhcp_ns = next((ns for ns in namespaces if ns.startswith("qdhcp-")), "")
    dhcp_ip = ""
    if dhcp_ns:
        addr_out = subprocess.run(f"sudo ip netns exec {dhcp_ns} ip addr show", shell=True,
                                  capture_output=True, text=True).stdout
        match = re.search(r"inet (10\.\d+\.\d+\.\d+)", addr_out)
        dhcp_ip = match.group(1) if match else ""
    print(f"  DHCP NS: {dhcp_ns}  DHCP IP: {dhcp_ip}")
    return vm_map, known_ids, dhcp_ns, dhcp_ip


def render_block(vm_map, known_ids, dhcp_ns, dhcp_ip):
    lines = ["VM_MAP = {"]
    for name, v in vm_map.items():
        lines.append(f'    "{name}":{" " * (17 - len(name))}{{"ip": "{v["ip"]}",'
                     f'{" " * (14 - len(v["ip"]))}"mac": "{v["mac"]}", "role": "{v["role"]}"}},')
    lines += ["}", "",
              'IP_TO_VM = {v["ip"]: k for k, v in VM_MAP.items()}',
              f'IP_TO_VM["{dhcp_ip or "10.0.0.2"}"] = "vm-app-02"  # DHCP namespace acts as attacker proxy',
              "", f'DHCP_NS = "{dhcp_ns}"', 'MIRROR_IFACE = "mirror0"']
    return "\n".join(lines)


def render_known_ids(known_ids):
    lines = ["KNOWN_IDS = {"]
    for name, ids in known_ids.items():
        lines += [f'    "{name}": {{', f'        "port_id": "{ids["port_id"]}",',
                  f'        "server_id": "{ids["server_id"]}",', "    },"]
    lines.append("}")
    return "\n".join(lines)


def main():
    vm_map, known_ids, dhcp_ns, dhcp_ip = discover()
    with open(CONSTANTS_PATH) as f:
        source = f.read()

    # replace only the two generated blocks; hand-written constants below them stay as they are
    source = re.sub(r"VM_MAP = \{.*?MIRROR_IFACE = \"mirror0\"",
                    render_block(vm_map, known_ids, dhcp_ns, dhcp_ip), source, flags=re.S)
    source = re.sub(r"KNOWN_IDS = \{.*?\n\}", render_known_ids(known_ids), source, flags=re.S)

    with open(CONSTANTS_PATH, "w") as f:
        f.write(source)
    print(f"\nConstants updated -> {CONSTANTS_PATH}")


if __name__ == "__main__":
    main()
