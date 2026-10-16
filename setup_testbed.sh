#!/bin/bash
# =============================================================================
# F-RIaaS Testbed Setup — Run on the DevStack host
# =============================================================================
# Creates 4 CirrOS VMs, configures security groups, and sets up OVS
# port mirroring. Must be run after DevStack is started.
#
# Usage:
#     source ~/devstack/openrc admin admin
#     bash setup_testbed.sh
# =============================================================================

set -e

echo "============================================================"
echo "  F-RIaaS Testbed Setup"
echo "============================================================"

# Check OpenStack is running
if ! openstack server list &>/dev/null; then
    echo "ERROR: OpenStack not reachable. Run: source ~/devstack/openrc admin admin"
    exit 1
fi

# ── Step 1: Create security group with permissive rules ─────────────
echo ""
echo "[1] Creating security group: friaas-allow-all ..."
openstack security group create friaas-allow-all \
    --description "FRIaaS testbed - allow all traffic" 2>/dev/null || true

# Allow all TCP, UDP, ICMP ingress
openstack security group rule create friaas-allow-all \
    --protocol tcp --dst-port 1:65535 --remote-ip 0.0.0.0/0 2>/dev/null || true
openstack security group rule create friaas-allow-all \
    --protocol udp --dst-port 1:65535 --remote-ip 0.0.0.0/0 2>/dev/null || true
openstack security group rule create friaas-allow-all \
    --protocol icmp --remote-ip 0.0.0.0/0 2>/dev/null || true

echo "  Security group configured"

# ── Step 2: Get CirrOS image and flavor ─────────────────────────────
echo ""
echo "[2] Checking CirrOS image and flavor ..."
IMAGE=$(openstack image list -f value -c Name | grep -i cirros | head -1)
if [ -z "$IMAGE" ]; then
    echo "  Downloading CirrOS image ..."
    wget -q http://download.cirros-cloud.net/0.6.2/cirros-0.6.2-x86_64-disk.img -O /tmp/cirros.img
    openstack image create "cirros-0.6.2" --file /tmp/cirros.img --disk-format qcow2 --container-format bare --public
    IMAGE="cirros-0.6.2"
fi
echo "  Image: $IMAGE"

# Check for m1.tiny flavor
FLAVOR=$(openstack flavor list -f value -c Name | grep -E "m1.tiny|cirros" | head -1)
if [ -z "$FLAVOR" ]; then
    openstack flavor create m1.tiny --ram 256 --disk 1 --vcpus 1
    FLAVOR="m1.tiny"
fi
echo "  Flavor: $FLAVOR"

# ── Step 3: Get or create network ───────────────────────────────────
echo ""
echo "[3] Checking network ..."
NETWORK=$(openstack network list --internal -f value -c Name | head -1)
if [ -z "$NETWORK" ]; then
    echo "  Creating private network ..."
    openstack network create private
    openstack subnet create private-subnet --network private \
        --subnet-range 10.0.0.0/24 --gateway 10.0.0.1 \
        --dns-nameserver 8.8.8.8
    NETWORK="private"
fi
echo "  Network: $NETWORK"

# ── Step 4: Create the 4 testbed VMs ───────────────────────────────
echo ""
echo "[4] Creating VMs ..."
for VM_NAME in vm-web-01 vm-app-02 vm-db-03 vm-monitor-04; do
    if openstack server show "$VM_NAME" &>/dev/null; then
        echo "  $VM_NAME already exists"
    else
        echo "  Creating $VM_NAME ..."
        openstack server create "$VM_NAME" \
            --image "$IMAGE" \
            --flavor "$FLAVOR" \
            --network "$NETWORK" \
            --security-group friaas-allow-all \
            --wait
    fi
done

# Wait for all VMs to be ACTIVE
echo ""
echo "  Waiting for VMs to become ACTIVE ..."
for i in $(seq 1 30); do
    ALL_ACTIVE=true
    for VM_NAME in vm-web-01 vm-app-02 vm-db-03 vm-monitor-04; do
        STATUS=$(openstack server show "$VM_NAME" -f value -c status 2>/dev/null)
        if [ "$STATUS" != "ACTIVE" ]; then
            ALL_ACTIVE=false
            break
        fi
    done
    if $ALL_ACTIVE; then
        echo "  All VMs are ACTIVE"
        break
    fi
    sleep 5
done

# Print VM details
echo ""
echo "  VM Details:"
openstack server list

# ── Step 5: Set up OVS port mirroring ──────────────────────────────
echo ""
echo "[5] Setting up OVS port mirroring ..."

# Create the mirror0 interface if it does not exist
if ! ip link show mirror0 &>/dev/null; then
    sudo ip link add mirror0 type dummy
fi
sudo ip link set mirror0 up
sudo ip link set mirror0 promisc on

# Add mirror0 to br-int as an OVS internal port
sudo ovs-vsctl --may-exist add-port br-int mirror0

# Get the port numbers for all VM tap interfaces
echo "  Finding VM tap interfaces ..."
MIRROR_PORTS=""
for VM_NAME in vm-web-01 vm-app-02 vm-db-03 vm-monitor-04; do
    PORT_ID=$(openstack port list --server "$VM_NAME" -f value -c ID 2>/dev/null | head -1)
    if [ -n "$PORT_ID" ]; then
        TAP="tap${PORT_ID:0:8}-${PORT_ID:8:2}"
        OVS_PORT=$(sudo ovs-vsctl get port "$TAP" _uuid 2>/dev/null || true)
        if [ -n "$OVS_PORT" ]; then
            MIRROR_PORTS="$MIRROR_PORTS $OVS_PORT"
            echo "    $VM_NAME -> $TAP ($OVS_PORT)"
        fi
    fi
done

# Get the mirror destination port UUID
MIRROR_DST=$(sudo ovs-vsctl get port mirror0 _uuid 2>/dev/null)
echo "  Mirror destination: mirror0 ($MIRROR_DST)"

# Create or update the mirror
if [ -n "$MIRROR_PORTS" ] && [ -n "$MIRROR_DST" ]; then
    # Remove existing mirror if any
    sudo ovs-vsctl -- --id=@m get mirror friaas_mirror -- remove bridge br-int mirrors @m 2>/dev/null || true

    # Build the mirror command
    MIRROR_CMD="sudo ovs-vsctl"
    PORT_REFS=""
    PORT_IDX=0
    for PORT_UUID in $MIRROR_PORTS; do
        MIRROR_CMD="$MIRROR_CMD -- --id=@p$PORT_IDX get port $PORT_UUID"
        PORT_REFS="$PORT_REFS @p$PORT_IDX"
        PORT_IDX=$((PORT_IDX + 1))
    done
    MIRROR_CMD="$MIRROR_CMD -- --id=@mirror0 get port mirror0"
    MIRROR_CMD="$MIRROR_CMD -- --id=@m create mirror name=friaas_mirror select-src-port=$PORT_REFS select-dst-port=$PORT_REFS output-port=@mirror0"
    MIRROR_CMD="$MIRROR_CMD -- add bridge br-int mirrors @m"

    eval $MIRROR_CMD
    echo "  OVS mirror configured successfully"
else
    echo "  WARNING: Could not configure mirror (missing ports)"
fi

# ── Step 6: Verify mirror is working ───────────────────────────────
echo ""
echo "[6] Verifying mirror ..."
sudo ovs-vsctl list mirror | grep -E "name|select|output"

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Testbed Setup Complete"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "    1. python3 update_vm_constants.py   (auto-discover IPs/MACs)"
echo "    2. python3 capture_dataset3.py   (capture real traffic)"
echo "    3. python3 run_all.py            (train models)"
echo "    4. python3 friaas_experiment.py  (run detection pipeline)"
echo ""
