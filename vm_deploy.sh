#!/bin/bash
set -e
echo "=== FRIaaS VM Deployment Script ==="
echo "Timestamp: $(date)"

echo "[1] Creating stack user..."
sudo useradd -s /bin/bash -d /opt/stack -m stack 2>/dev/null || echo "stack user exists"
echo "stack ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/stack
sudo chmod 440 /etc/sudoers.d/stack

echo "[2] Cloning DevStack stable/2025.1..."
sudo -u stack bash -c '
cd /opt/stack
rm -rf devstack
git clone https://opendev.org/openstack/devstack -b stable/2025.1
echo "DevStack cloned"
'

echo "[3] Copying local.conf..."
sudo cp ~/friaas-prototype/vm-config/local.conf /opt/stack/devstack/local.conf
sudo chown stack:stack /opt/stack/devstack/local.conf

echo "[4] Installing Python requirements..."
cd ~/friaas-prototype
pip3 install --user -r requirements.txt 2>&1 | tail -5

echo "[5] Applying netplan..."
sudo cp ~/friaas-prototype/vm-config/01-netcfg.yaml /etc/netplan/01-netcfg.yaml
sudo chmod 600 /etc/netplan/01-netcfg.yaml
sudo netplan apply 2>&1 || echo "netplan apply had warnings"

echo "=== Pre-DevStack setup complete ==="
echo "Next: sudo su - stack && cd devstack && ./stack.sh"
