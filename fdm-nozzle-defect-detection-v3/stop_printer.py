"""Standalone printer stop command demo."""

from src.fdm_defect_detection.printer_control import mock_stop_printer, send_http_stop, send_serial_stop


if __name__ == "__main__":
    mock_stop_printer()
    # send_serial_stop(port="/dev/ttyUSB0", baudrate=115200)
    # send_http_stop(url="http://127.0.0.1:7125/printer/gcode/script", payload={"script": "M112"})
