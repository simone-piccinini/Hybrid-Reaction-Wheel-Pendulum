#!/usr/bin/env python3
"""Minimal serial terminal that ALWAYS writes the log file.

PuTTY's session logging is easy to misconfigure and fails silently when the
folder does not exist or the settings were applied to the wrong session. This
does the same job with no GUI: it shows the stream on screen, writes every line
to a file, and flushes after each one, so the file is complete even if the
program is killed.

Usage
-----
    python scripts/serial_log.py COM3 data/hw/s0c_wobble_180deg.log
    python scripts/serial_log.py COM3 data/hw/out.log --baud 115200

Type commands and press Enter, exactly as in any terminal. Ctrl+C to quit.

The parent directory is created if missing. The baud rate is irrelevant on a
Teensy USB CDC port but is accepted for other boards.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial is missing.  pip install pyserial")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port", help="e.g. COM3, or /dev/ttyACM0")
    ap.add_argument("logfile")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--append", action="store_true",
                    help="append instead of overwriting")
    a = ap.parse_args()

    path = Path(a.logfile)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ser = serial.Serial(a.port, a.baud, timeout=0.2)
    except serial.SerialException as e:
        sys.exit(f"cannot open {a.port}: {e}\n"
                 "Is the Arduino Serial Monitor or PuTTY still holding it?")

    fh = path.open("a" if a.append else "w", encoding="utf-8", newline="\n")
    stop = threading.Event()
    lines = 0

    def reader() -> None:
        nonlocal lines
        while not stop.is_set():
            try:
                raw = ser.readline()
            except Exception:
                break
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            print(text, flush=True)
            fh.write(text + "\n")
            fh.flush()          # the whole point: never lose the tail
            lines += 1

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    print(f"# logging {a.port} -> {path.resolve()}")
    print("# type commands and press Enter.  Ctrl+C to stop.\n")
    try:
        for line in sys.stdin:
            ser.write((line.rstrip("\r\n") + "\n").encode())
            ser.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        t.join(timeout=1.0)
        fh.close()
        ser.close()
        print(f"\n# wrote {lines} lines to {path.resolve()}")


if __name__ == "__main__":
    main()
