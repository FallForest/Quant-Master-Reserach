#!/usr/bin/env python
"""TDX Binary Protocol SSL Proxy — captures raw trading traffic.

Placed between xiadan.exe and the real trading server so we can
observe the exact binary frame format.

Architecture:
    xiadan  ──SSL──>  proxy:16002  ──SSL──>  real-server:6002
                          │
                     capture.log  (raw hex + ASCII dump)

Usage:
    1. python scripts/tdx_ssl_proxy.py        # starts proxy, waits for xiadan
    2. Launch xiadan.exe and log in
    3. Perform trading operations
    4. Ctrl-C to stop, check capture.log
"""

import os
import ssl
import socket
import struct
import threading
import time
from pathlib import Path

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 16002

REAL_HOST = "1.202.143.39"
REAL_PORT = 6002

LOG_FILE = Path(__file__).parent / "tdx_capture.log"
CERT_FILE = Path(__file__).parent / "proxy_cert.pem"
KEY_FILE = Path(__file__).parent / "proxy_key.pem"


def generate_self_signed_cert():
    """Generate a self-signed cert if none exists."""
    if CERT_FILE.exists() and KEY_FILE.exists():
        return

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    with open(CERT_FILE, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(KEY_FILE, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    print(f"[+] Generated self-signed cert: {CERT_FILE}")


def hexdump(data: bytes, prefix: str = "") -> str:
    """Format bytes as hex + ASCII dump."""
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{prefix}{i:08x}  {hex_part:<48s}  {ascii_part}")
    return "\n".join(lines)


class CaptureLogger:
    """Thread-safe logger for captured data."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        # Clear previous log
        self.path.write_text("")

    def log(self, direction: str, data: bytes):
        ts = time.strftime("%H:%M:%S.%f")[:-3]
        with self.lock:
            with open(self.path, "a") as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"[{ts}] {direction} ({len(data)} bytes)\n")
                f.write(hexdump(data, "  ") + "\n")
                if len(data) >= 4:
                    # Try to parse frame header
                    f.write(f"  First 4 bytes: {data[:4].hex()}\n")
                f.flush()


logger = CaptureLogger(LOG_FILE)


def relay(src, dst, label: str):
    """Relay data from src to dst, logging each chunk."""
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            logger.log(label, data)
            dst.sendall(data)
    except Exception as e:
        pass
    finally:
        try:
            src.shutdown(socket.SHUT_RD)
        except:
            pass
        try:
            dst.shutdown(socket.SHUT_WR)
        except:
            pass


def handle_xiadan(conn_sock):
    """Handle one xiadan connection: accept SSL, connect to real server, relay."""
    peer = conn_sock.getpeername()
    print(f"[+] xiadan connected from {peer}")

    # --- SSL to real server ---
    ctx_client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx_client.check_hostname = False
    ctx_client.verify_mode = ssl.CERT_NONE

    try:
        raw_server = socket.create_connection((REAL_HOST, REAL_PORT), timeout=10)
        server_ssl = ctx_client.wrap_socket(raw_server, server_hostname=REAL_HOST)
        print(f"[+] Connected to real server {REAL_HOST}:{REAL_PORT} (SSL)")
    except Exception as e:
        print(f"[-] Cannot connect to real server: {e}")
        conn_sock.close()
        return

    # --- SSL to xiadan (we act as server) ---
    ctx_server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx_server.load_cert_chain(str(CERT_FILE), str(KEY_FILE))

    try:
        xiadan_ssl = ctx_server.wrap_socket(conn_sock, server_side=True)
        print(f"[+] SSL handshake with xiadan OK")
    except ssl.SSLError as e:
        print(f"[-] SSL handshake with xiadan FAILED: {e}")
        print("    xiadan may not accept self-signed certs.")
        # Try plain TCP fallback
        print("[*] Trying plain TCP (no SSL) to xiadan...")
        xiadan_ssl = conn_sock  # raw socket

    # --- Bidirectional relay ---
    t1 = threading.Thread(target=relay, args=(xiadan_ssl, server_ssl, "C->S"), daemon=True)
    t2 = threading.Thread(target=relay, args=(server_ssl, xiadan_ssl, "S->C"), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    xiadan_ssl.close()
    server_ssl.close()
    print(f"[-] Connection from {peer} closed")


def main():
    generate_self_signed_cert()

    # Listen for xiadan
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((PROXY_HOST, PROXY_PORT))
    server_sock.listen(5)
    print(f"[*] TDX SSL Proxy listening on {PROXY_HOST}:{PROXY_PORT}")
    print(f"[*] Forwarding to {REAL_HOST}:{REAL_PORT}")
    print(f"[*] Capture log: {LOG_FILE}")
    print(f"[*] Start xiadan.exe now, then perform trading operations.")
    print(f"[*] Press Ctrl-C to stop.\n")

    try:
        while True:
            conn, addr = server_sock.accept()
            t = threading.Thread(target=handle_xiadan, args=(conn,), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print(f"\n[*] Stopped. Check {LOG_FILE} for captured data.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
