"""Cloud platform API calls: suspect VM lookup, quarantine, restore, infrastructure logs."""
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import openstack

from services.constants import IP_TO_VM

QUARANTINE_SG = "friaas-quarantine"
ALLOW_ALL_SG = "friaas-allow-all"
OPENRC = ". /opt/stack/devstack/openrc admin admin"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _get_connection():
    """openstacksdk connection using the DevStack admin credentials unless overridden."""
    os.environ.setdefault("OS_AUTH_URL", "http://192.168.100.1/identity")
    os.environ.setdefault("OS_PROJECT_NAME", "admin")
    os.environ.setdefault("OS_USERNAME", "admin")
    os.environ.setdefault("OS_PASSWORD", "secret")
    os.environ.setdefault("OS_USER_DOMAIN_NAME", "Default")
    os.environ.setdefault("OS_PROJECT_DOMAIN_NAME", "Default")
    return openstack.connect()


def identify_vm_by_ip(suspect_ip):
    """Identify Suspect VM: the networking service records every virtual port
    with its fixed IPs and owning instance, so the port holding the IP names the VM."""
    try:
        conn = _get_connection()
        for port in conn.network.ports():
            if not (port.device_owner or "").startswith("compute:"):
                continue
            if any(fixed.get("ip_address") == suspect_ip for fixed in (port.fixed_ips or [])):
                return conn.compute.get_server(port.device_id).name, port.mac_address
    except Exception as exc:
        print(f"    Port lookup via SDK failed ({exc}), using static IP map")
    return IP_TO_VM.get(suspect_ip, "unknown"), None


def _openstack_cli(command):
    return subprocess.run(f"{OPENRC} && {command}", shell=True, executable="/bin/bash",
                          capture_output=True, text=True, timeout=30)


def isolate_vm(vm_name):
    """Apply Quarantine Security Group: swap every port of the VM onto a
    rule-less group and remember the original groups so the VM can be restored."""
    print(f"  [Isolate] Isolating {vm_name} via openstacksdk ...")
    t0 = time.perf_counter()
    try:
        conn = _get_connection()
        server = conn.compute.find_server(vm_name)
        if not server:
            print(f"    ERROR: Server {vm_name} not found")
            return None

        sg = conn.network.find_security_group(QUARANTINE_SG)
        if not sg:
            sg = conn.network.create_security_group(
                name=QUARANTINE_SG, description="F-RIaaS quarantine group (deny all)")
            # Neutron adds default egress rules to a new group a quarantine must have none
            for rule in conn.network.security_group_rules(security_group_id=sg.id):
                conn.network.delete_security_group_rule(rule.id)

        ports = list(conn.network.ports(device_id=server.id))
        original_sgs = {port.id: port.security_group_ids for port in ports}
        for port in ports:
            conn.network.update_port(port.id, security_groups=[sg.id])

        print(f"    {vm_name} isolated with {QUARANTINE_SG}")
        return {
            "vm_id": vm_name,
            "server_id": server.id,
            "security_group": QUARANTINE_SG,
            "security_group_id": sg.id,
            "original_security_groups": original_sgs,
            "timestamp": _utc_now(),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "action": f"Applied deny-all security group to {vm_name}",
            "status": "isolated",
        }
    except Exception as exc:
        print(f"    Isolation via SDK failed ({exc}), falling back to CLI")
        return _isolate_vm_cli(vm_name, t0)


def _isolate_vm_cli(vm_name, t0):
    sg_name = f"sg-isolate-{vm_name}"
    _openstack_cli(f"openstack security group create {sg_name} "
                   f"--description 'Isolation group for {vm_name}' 2>/dev/null")
    _openstack_cli(f"for rule_id in $(openstack security group rule list {sg_name} "
                   f"-f value -c ID); do openstack security group rule delete $rule_id "
                   f"2>/dev/null; done")
    _openstack_cli(f"openstack server remove security group {vm_name} {ALLOW_ALL_SG} 2>/dev/null")
    _openstack_cli(f"openstack server add security group {vm_name} {sg_name} 2>/dev/null")
    return {
        "vm_id": vm_name,
        "security_group": sg_name,
        "timestamp": _utc_now(),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "action": f"Applied deny-all security group to {vm_name}",
        "status": "isolated",
    }


def restore_vm(vm_name, sg_name=QUARANTINE_SG, original_security_groups=None):
    """Restore Security Group: put back the groups recorded at quarantine,
    or the testbed allow-all group when no record exists."""
    print(f"  [Restore] Restoring {vm_name} ...")
    original_security_groups = original_security_groups or {}
    try:
        conn = _get_connection()
        server = conn.compute.find_server(vm_name)
        if server:
            allow_all = conn.network.find_security_group(ALLOW_ALL_SG)
            fallback = [allow_all.id] if allow_all else None
            for port in conn.network.ports(device_id=server.id):
                groups = original_security_groups.get(port.id) or fallback
                # an empty group list would block the port completely, so leave it alone
                if groups:
                    conn.network.update_port(port.id, security_groups=groups)
            print(f"    {vm_name} restored to normal operation")
            return
    except Exception as exc:
        print(f"    SDK restore failed ({exc}), falling back to CLI")

    _openstack_cli(f"openstack server remove security group {vm_name} {sg_name} 2>/dev/null && "
                   f"openstack server add security group {vm_name} {ALLOW_ALL_SG} 2>/dev/null")
    print(f"    {vm_name} restored to normal operation")


def collect_infrastructure_logs(vm_name):
    """Infrastructure-level logs for Log Collection: the port records of the
    networking service and the instance action records of the compute service."""
    print(f"  [Infra] Collecting infrastructure logs for {vm_name} ...")
    infra = {"neutron_port_events": [], "nova_instance_actions": []}
    try:
        conn = _get_connection()
        server = conn.compute.find_server(vm_name)
        if not server:
            return infra

        for port in conn.network.ports(device_id=server.id):
            infra["neutron_port_events"].append({
                "port_id": port.id,
                "mac_address": port.mac_address,
                "status": port.status,
                "security_groups": port.security_group_ids,
                "fixed_ips": [{"ip": ip["ip_address"], "subnet": ip["subnet_id"]}
                              for ip in (port.fixed_ips or [])],
                "updated_at": str(port.updated_at) if port.updated_at else "",
            })

        # The SDK has no instance-action listing on this release, so the CLI is used here
        result = _openstack_cli(f"openstack server event list {vm_name} -f json 2>/dev/null")
        if result.returncode == 0 and result.stdout.strip():
            for event in json.loads(result.stdout):
                infra["nova_instance_actions"].append({
                    "action": event.get("Action", ""),
                    "request_id": event.get("Request ID", ""),
                    "start_time": event.get("Start Time", ""),
                })

        print(f"    Collected {len(infra['neutron_port_events'])} port event(s), "
              f"{len(infra['nova_instance_actions'])} Nova action(s)")
    except Exception as exc:
        print(f"    Infrastructure log collection failed: {exc}")
    return infra
