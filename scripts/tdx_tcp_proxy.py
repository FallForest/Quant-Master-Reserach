#!/usr/bin/env python
"""Plain TCP proxy for TDX binary protocol capture.

xiadan is configured for plaintext HEXIN on 127.0.0.1:16002,
this proxy relays to the real server 1.202.143.39:6002 and logs
all raw bytes.

Usage:
    python scripts/tdx_tcp_proxy.py
"""

import socket
import threading
import time
from pathlib import Path

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 16002

REAL_HOST = "1.202.143.39"
REAL_PORT = 6002

LOG_FILE = Path(__file__).parent / "tdx_capture.log"


def hexdump(data: bytes, prefix: str = "") -> str:
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{prefix}{i:08x}  {hex_part:<48s}  {ascii_part}")
    return "\n".join(lines)


class CaptureLogger:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.path.write_text("")

    def log(self, direction: str, data: bytes):
        ts = time.strftime("%H:%M:%S.%f")[:-3]
        with self.lock:
            with open(self.path, "a") as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"[{ts}] {direction} ({len(data)} bytes)\n")
                f.write(hexdump(data, "  ") + "\n")
                if len(data) >= 4:
                    f.write(f"  Header bytes: {data[:4].hex()}")
                    if len(data) >= 8:
                        f.write(f"  Next 4: {data[4:8].hex()}")
                    f.write("\n")
                f.flush()


logger = CaptureLogger(LOG_FILE)


def relay(src, dst, label: str):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            logger.log(label, data)
            print(f"  [{label}] {len(data)} bytes")
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try:
            src.shutdown(socket.SHUT_RD)
        except Exception:
            pass
        try:
            dst.shutdown(socket.SHUT_WR)
        except Exception:
            pass


def handle_client(conn_sock):
    peer = conn_sock.getpeername()
    print(f"[+] xiadan connected from {peer}")

    try:
        real_sock = socket.create_connection((REAL_HOST, REAL_PORT), timeout=30)
        print(f"[+] Connected to real server {REAL_HOST}:{REAL_PORT}")
    except Exception as e:
        print(f"[-] Cannot connect to real server: {e}")
        conn_sock.close()
        return

    t1 = threading.Thread(target=relay, args=(conn_sock, real_sock, "C->S"), daemon=True)
    t2 = threading.Thread(target=relay, args=(real_sock, conn_sock, "S->C"), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    conn_sock.close()
    real_sock.close()
    print(f"[-] Connection from {peer} closed")


def main():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((PROXY_HOST, PROXY_PORT))
    server_sock.listen(5)
    print(f"[*] TDX TCP Proxy listening on {PROXY_HOST}:{PROXY_PORT}")
    print(f"[*] Forwarding to {REAL_HOST}:{REAL_PORT}")
    print(f"[*] Capture log: {LOG_FILE}")
    print(f"[*] Start xiadan now, then perform trading operations.")
    print(f"[*] Press Ctrl-C to stop.\n")

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"[*] Accepted connection from {addr}")
            t = threading.Thread(target=handle_client, args=(conn,), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print(f"\n[*] Stopped. Check {LOG_FILE} for captured data.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
