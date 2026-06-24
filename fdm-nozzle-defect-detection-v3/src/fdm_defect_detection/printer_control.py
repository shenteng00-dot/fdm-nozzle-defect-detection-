from __future__ import annotations

import json
from typing import Dict, Optional


def mock_stop_printer(reason: str = "high_risk_nozzle_defect") -> None:
    """Mock emergency stop for demo and interview presentation."""
    print(f"[MOCK STOP] Emergency stop signal generated. reason={reason}")


def send_serial_stop(port: str = "/dev/ttyUSB0", baudrate: int = 115200) -> None:
    """Send emergency stop G-code by serial.

    M112 is a common emergency stop command for many 3D printer firmwares.
    """
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required: pip install pyserial") from exc

    with serial.Serial(port, baudrate, timeout=1) as ser:
        ser.write(b"M112\n")
        ser.flush()


def send_http_stop(url: str, payload: Optional[Dict[str, object]] = None) -> None:
    """Send stop request to a printer control server, e.g. OctoPrint or Moonraker."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required: pip install requests") from exc

    data = payload or {"command": "M112"}
    response = requests.post(url, json=data, timeout=3)
    response.raise_for_status()
    print("[HTTP STOP] response=", json.dumps(response.json() if response.text else {}, ensure_ascii=False))
