# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""Helpers for creating broker implementations with safer defaults.

The project currently contains multiple live-trading adapters with very
different reliability characteristics:

- ``paper``: in-memory broker for dry runs and workflow tests
- ``tcdll``: Tc.dll helper process, preferred non-easytrader live route
- ``tdx``: HTTP/TQL adapter, usable for endpoints that expose TQLEX trading
- ``easytrader``: optional GUI automation wrapper
- ``xiadan``: direct Win32 control, kept as an experimental fallback only

This factory centralizes broker selection so new code does not accidentally
default to the brittle direct-window automation path.
"""

from typing import Any

from .base import BaseBroker
from .paper_broker import PaperBroker
from .tcdll_broker import TcDllBroker
from .tdx.broker import TDXBroker
from .xiadan_broker import XiadanBroker


def create_broker(kind: str = "paper", **kwargs: Any) -> BaseBroker:
    """Create a broker instance with explicit routing.

    Parameters
    ----------
    kind : str
        One of ``paper``, ``tcdll``, ``tdx``, ``easytrader``, or ``xiadan``.
        ``paper`` is the default so command-line tooling is safe by default.
    kwargs : dict
        Extra keyword arguments passed to the underlying broker constructor.
    """
    normalized = kind.strip().lower()

    if normalized in {"paper", "mock", "dry_run", "dry-run"}:
        return PaperBroker(**kwargs)

    if normalized in {"tcdll", "tc_dll", "tc", "dll"}:
        return TcDllBroker(**kwargs)

    if normalized in {"easytrader", "easy_trader", "yh", "yh_client"}:
        from .easytrader_broker import EasytraderBroker

        broker_type = kwargs.pop("broker_type", "yh_client")
        return EasytraderBroker(broker_type=broker_type)

    if normalized == "tdx":
        host = kwargs.pop("host", None)
        if not host:
            raise ValueError("TDX broker requires 'host'")
        return TDXBroker(host=host, **kwargs)

    if normalized in {"xiadan", "win32", "direct"}:
        return XiadanBroker()

    raise ValueError(
        f"Unsupported broker kind '{kind}'. Expected one of: paper, tcdll, tdx, easytrader, xiadan."
    )
