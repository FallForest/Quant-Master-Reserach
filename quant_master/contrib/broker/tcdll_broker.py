# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""Broker adapter for the bundled 32-bit Tc.dll helper process.

The helper executable wraps ``Tc.dll`` and is the non-easytrader route for
brokers that expose trading through the xiadan/TDX binary interface. Live
order submission is opt-in via ``enable_live=True`` so connectivity tests and
dry runs cannot accidentally place orders.
"""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional

from quant_master.log import get_module_logger

from .base import AccountInfo, BaseBroker, BrokerOrder, BrokerOrderDir, OrderStatus, Position
from .tdx.consts import TDX_ENTRIES
from .tdx.protocol import build_buy_params, build_sell_params, infer_market

logger = get_module_logger("TcDllBroker")


class TcDllBroker(BaseBroker):
    """BaseBroker implementation backed by scripts/tdx_caller.exe."""

    def __init__(
        self,
        caller_path: Optional[str] = None,
        *,
        enable_live: bool = False,
        startup_timeout: float = 10.0,
        command_timeout: float = 15.0,
        xiadan_work_dir: Optional[str] = None,
        dll_dir: Optional[str] = None,
    ):
        root = Path(__file__).resolve().parents[3]
        self.caller_path = self._absolute_path(caller_path) if caller_path else root / "scripts" / "tdx_caller.exe"
        self.xiadan_work_dir = self._absolute_path(xiadan_work_dir) if xiadan_work_dir else self.caller_path.parent
        self.dll_dir = self._absolute_path(dll_dir) if dll_dir else self.xiadan_work_dir
        self.enable_live = enable_live
        self.startup_timeout = startup_timeout
        self.command_timeout = command_timeout
        self._proc: Optional[subprocess.Popen[str]] = None
        self._connected = False
        self._stdout_queue: "Queue[Optional[str]]" = Queue()
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._reader_threads: List[threading.Thread] = []
        self._last_command: Optional[str] = None
        self._last_returncode: Optional[int] = None

    def connect(self) -> bool:
        if self._connected:
            return True
        if not self.caller_path.exists():
            raise FileNotFoundError(f"tdx_caller.exe not found: {self.caller_path}")
        if not self.xiadan_work_dir.exists():
            raise FileNotFoundError(f"xiadan work directory not found: {self.xiadan_work_dir}")

        args = [str(self.caller_path)]
        if self.xiadan_work_dir:
            args.extend(["--work-dir", str(self.xiadan_work_dir)])
        if self.dll_dir:
            args.extend(["--dll-dir", str(self.dll_dir)])
        if self.enable_live:
            args.append("--allow-live")

        self._proc = subprocess.Popen(
            args,
            cwd=str(self.xiadan_work_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._start_readers()
        ready = self._readline(timeout=self.startup_timeout)
        if ready != "READY":
            self.close()
            raise RuntimeError(f"tdx_caller did not become ready: {ready}")
        self._connected = True
        return True

    def login(self, account: str, password: str, **kwargs) -> bool:
        self.connect()
        extra = kwargs.get("extra", "")
        line = self._command(f"LOGIN {account} {password} {extra}".strip())
        return line.startswith("OK LOGIN=") and not line.endswith("=-1")

    def logout(self) -> None:
        self.close()

    def buy(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        return self._submit(stock_id, price, amount, BrokerOrderDir.BUY, dry_run=kwargs.get("dry_run", False))

    def sell(self, stock_id: str, price: float, amount: int, **kwargs) -> BrokerOrder:
        return self._submit(stock_id, price, amount, BrokerOrderDir.SELL, dry_run=kwargs.get("dry_run", False))

    def cancel_order(self, order_id: str) -> bool:
        if not self.enable_live:
            raise RuntimeError("TcDllBroker cancel_order requires enable_live=True")
        self.connect()
        line = self._command(f"GRIDJY {TDX_ENTRIES['cancel_order']} Wth={order_id}|")
        return line.startswith("OK GRIDJY=") and not line.endswith("=-1")

    def query_orders(self) -> List[BrokerOrder]:
        return []

    def query_deals(self) -> List[dict]:
        return []

    def query_positions(self) -> List[Position]:
        return []

    def query_account(self) -> AccountInfo:
        return AccountInfo(total_assets=0.0, available_cash=0.0, market_value=0.0, frozen_amount=0.0)

    def is_connected(self) -> bool:
        return self._connected and self._proc is not None and self._proc.poll() is None

    def version(self) -> str:
        return self.safe_probe("version")

    def probe(self, probe: str, *, gridjy_entry: str = "DRYRUN", gridjy_params: str = "") -> Dict[str, object]:
        normalized = probe.lower().replace("-", "_")
        if normalized == "gridjy_dryrun":
            line = self.gridjy_dryrun(gridjy_entry, gridjy_params)
        else:
            line = self.safe_probe(normalized)
        payload = self.parse_probe_line(line)
        payload["probe"] = normalized
        return payload

    def status(self) -> str:
        return self.safe_probe("status")

    def diagnostics(self) -> dict:
        proc = self._proc
        return {
            "caller_path": str(self.caller_path),
            "xiadan_work_dir": str(self.xiadan_work_dir),
            "dll_dir": str(self.dll_dir),
            "connected": self.is_connected(),
            "returncode": self._poll_returncode() if proc is not None else self._last_returncode,
            "last_command": self._last_command,
            "stderr_tail": self._recent_stderr(),
        }

    def safe_probe(self, probe: str) -> str:
        """Run a non-trading helper probe.

        The allowlist intentionally excludes GRIDJY/LEVINJY/OPERATE so dry
        runs and diagnostics cannot invoke Tc.dll trading functions.
        """
        commands = {
            "ping": "PING",
            "functions": "FUNCTIONS",
            "version": "VERSION",
            "status": "STATUS",
            "status_raw": "STATUS_RAW",
            "createall": "CREATEALL",
        }
        try:
            command = commands[probe.lower()]
        except KeyError as exc:
            allowed = ", ".join(sorted(commands))
            raise ValueError(f"Unsupported safe Tc.dll probe: {probe!r}. Allowed: {allowed}") from exc
        self.connect()
        return self._command(command)

    def gridjy_dryrun(self, entry: str, params: str = "") -> str:
        self.connect()
        return self._command(f"GRIDJY_DRYRUN {entry} {params}".rstrip())

    @staticmethod
    def parse_probe_line(line: str) -> Dict[str, object]:
        """Convert one tdx_caller response line into a stable JSON-friendly dict."""
        text = line.strip()
        parts = text.split()
        status = parts[0] if parts else ""
        payload: Dict[str, object] = {
            "ok": status == "OK",
            "status": status,
            "raw": text,
        }
        if len(parts) > 1:
            payload["message"] = " ".join(parts[1:])
        for token in parts[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            payload[key.lower()] = TcDllBroker._coerce_probe_value(value)
        return payload

    @staticmethod
    def _coerce_probe_value(value: str) -> object:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    def close(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                self._write("QUIT")
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        self._proc = None
        self._connected = False

    def _submit(self, stock_id: str, price: float, amount: int, direction: BrokerOrderDir, *, dry_run: bool) -> BrokerOrder:
        if dry_run:
            return BrokerOrder(stock_id=stock_id, price=price, amount=amount, direction=direction, order_id="DRY-RUN")
        if not self.enable_live:
            raise RuntimeError("TcDllBroker live trading requires enable_live=True")

        market = infer_market(stock_id)
        if direction == BrokerOrderDir.BUY:
            entry = TDX_ENTRIES["buy"]
            params = build_buy_params(stock_id, market, price, amount)
        else:
            entry = TDX_ENTRIES["sell"]
            params = build_sell_params(stock_id, market, price, amount)
        payload = "|".join(f"{key}={value}" for key, value in params.items()) + "|"

        self.connect()
        line = self._command(f"GRIDJY {entry} {payload}")
        if not line.startswith("OK GRIDJY=") or line.endswith("=-1"):
            raise RuntimeError(f"Tc.dll order failed: {line}")
        return BrokerOrder(
            stock_id=stock_id,
            price=price,
            amount=amount,
            direction=direction,
            order_id=line.split("=", 1)[-1],
            status=OrderStatus.PENDING,
        )

    def _command(self, line: str, *, timeout: Optional[float] = None) -> str:
        if self._is_live_helper_command(line) and not self.enable_live:
            raise RuntimeError(f"TcDllBroker refuses live helper command without enable_live=True: {line.split()[0]}")
        self._last_command = line
        self._write(line)
        return self._readline(timeout=timeout)

    def _write(self, line: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("tdx_caller is not connected")
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _readline(self, *, timeout: Optional[float] = None) -> str:
        if self._proc is None:
            raise RuntimeError("tdx_caller is not connected")
        timeout = self.command_timeout if timeout is None else timeout
        try:
            line = self._stdout_queue.get(timeout=timeout)
        except Empty as exc:
            rc = self._poll_returncode()
            stderr = self._recent_stderr()
            if rc is None:
                raise TimeoutError(
                    f"tdx_caller timed out after {timeout:.1f}s "
                    f"while waiting for {self._last_command!r}. stderr_tail={stderr!r}"
                ) from exc
            raise RuntimeError(
                f"tdx_caller exited with code {rc} while waiting for {self._last_command!r}. "
                f"stderr_tail={stderr!r}"
            ) from exc
        if line is None:
            rc = self._poll_returncode(wait=0.25)
            stderr = self._recent_stderr()
            raise RuntimeError(
                f"tdx_caller closed unexpectedly (code={rc}) while waiting for {self._last_command!r}. "
                f"stderr_tail={stderr!r}"
            )
        return line.strip()

    def _start_readers(self) -> None:
        self._stdout_queue = Queue()
        self._stderr_lines = deque(maxlen=200)
        self._reader_threads = []
        if self._proc is None:
            return
        if self._proc.stdout is not None:
            thread = threading.Thread(target=self._drain_stdout, args=(self._proc.stdout,), daemon=True)
            thread.start()
            self._reader_threads.append(thread)
        if self._proc.stderr is not None:
            thread = threading.Thread(target=self._drain_stderr, args=(self._proc.stderr,), daemon=True)
            thread.start()
            self._reader_threads.append(thread)

    def _drain_stdout(self, stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                self._stdout_queue.put(line)
        finally:
            self._stdout_queue.put(None)

    def _drain_stderr(self, stream) -> None:
        for line in iter(stream.readline, ""):
            text = line.rstrip()
            self._stderr_lines.append(text)
            logger.debug("tdx_caller stderr: %s", text)

    def _recent_stderr(self) -> str:
        return "\n".join(self._stderr_lines)

    def _poll_returncode(self, *, wait: float = 0.0) -> Optional[int]:
        if self._proc is None:
            return self._last_returncode
        rc = self._proc.poll()
        if rc is None and wait > 0:
            try:
                rc = self._proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                rc = self._proc.poll()
        if rc is not None:
            self._last_returncode = rc
        return rc

    @staticmethod
    def _is_live_helper_command(line: str) -> bool:
        command = line.split(maxsplit=1)[0].upper() if line.strip() else ""
        return command in {"GRIDJY", "LEVINJY", "OPERATE"}

    @staticmethod
    def _absolute_path(value: str) -> Path:
        return Path(value).expanduser().resolve(strict=False)

    def __enter__(self) -> "TcDllBroker":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
