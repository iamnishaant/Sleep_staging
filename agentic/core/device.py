"""Device health for the resource gate before the report model runs (policy `resources.*`).

SoC temperature is read on Linux (Raspberry Pi: /sys/class/thermal); elsewhere it is None (no sensor),
and only RAM is checked.
"""
from __future__ import annotations

from pathlib import Path

THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")


def device_health() -> dict:
    import psutil
    temp = int(THERMAL.read_text().strip()) / 1000 if THERMAL.exists() else None
    return {"soc_temp_c": temp, "free_ram_gb": round(psutil.virtual_memory().available / 1e9, 2)}
