"""Restore vm-app-02 from quarantine."""
from services.openstack_service import restore_vm
restore_vm("vm-app-02", "friaas-quarantine", ["friaas-allow-all"])
print("vm-app-02 restored to friaas-allow-all")
