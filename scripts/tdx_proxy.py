#!/usr/bin/env python
"""TDX Traffic Capture Proxy v2.

Properly handles HTTP requests with binary bodies.
Captures and logs full request/response including encrypted payloads.

Usage:
    1. Modify connect.cfg: TPHost01=127.0.0.1, TPHost02=127.0.0.1
    2. Run this script as administrator
    3. Restart TdxW
    4. All traffic logged to tdx_capture.log
    5. Ctrl+C to stop, then restore connect.cfg
"""

import socket
import threading
import os
import sys
import binascii
from datetime import datetime

REAL_HOST = "36.110.86.110"
REAL_PORT = 7615
LOCAL_PORT = 7615

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdx_capture.log")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_hex(label, data):
    """Log data as hex dump with ASCII side panel."""
    if not data:
        return
    hex_str = binascii.hexlify(data).decode()
    # Log first 256 bytes as hex
    preview = hex_str[:512]
    log(f"{label} ({len(data)} bytes): {preview}")
    # Also try to extract readable strings
    readable = []
    current = []
    for b in data:
        if 32 <= b < 127:
            current.append(chr(b))
        else:
            if len(current) >= 3:
                readable.append("".join(current))
            current = []
    if len(current) >= 3:
        readable.append("".join(current))
    if readable:
        log(f"{label} TEXT: {' | '.join(readable)}")


def handle_connection(client_sock, addr):
    conn_id = f"{addr[0]}:{addr[1]}"
    log(f"\n{'='*60}")
    log(f"NEW CONNECTION: {conn_id}")
    log(f"{'='*60}")

    try:
        server_sock = socket.create_connection((REAL_HOST, REAL_PORT), timeout=30)
        log(f"Connected to {REAL_HOST}:{REAL_PORT}")
    except Exception as e:
        log(f"ERROR: Cannot connect to real server: {e}")
        client_sock.close()
        return

    def forward(src, dst, direction):
        try:
            buf = b""
            while True:
                try:
                    chunk = src.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                except socket.timeout:
                    if buf:
                        log_hex(f"{direction}", buf)
                        dst.sendall(buf)
                        buf = b""
                    continue

                if buf:
                    log_hex(f"{direction}", buf)
                    dst.sendall(buf)
                    buf = b""
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            log(f"Forward error ({direction}): {e}")

    t1 = threading.Thread(target=forward, args=(client_sock, server_sock, ">>> REQUEST"), daemon=True)
    t2 = threading.Thread(target=forward, args=(server_sock, client_sock, "<<< RESPONSE"), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=300)

    client_sock.close()
    server_sock.close()
    log(f"Connection {conn_id} closed\n")


def main():
    print("=" * 60)
    print("  TDX Traffic Capture Proxy v2")
    print("=" * 60)
    print(f"  Forwarding: localhost:{LOCAL_PORT} -> {REAL_HOST}:{REAL_PORT}")
    print(f"  Log: {LOG_FILE}")
    print()

    with open(LOG_FILE, "w") as f:
        f.write(f"TDX Capture v2 started at {datetime.now()}\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", LOCAL_PORT))
    sock.listen(5)
    log(f"Proxy listening on 127.0.0.1:{LOCAL_PORT}")
    log("Waiting for connections...\n")

    try:
        while True:
            csock, caddr = sock.accept()
            t = threading.Thread(target=handle_connection, args=(csock, caddr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        log("\nShutting down.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
