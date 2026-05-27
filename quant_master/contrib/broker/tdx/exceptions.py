# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.


class TDXError(Exception):
    """Base exception for TDX operations."""


class TDXConnectionError(TDXError):
    """Connection to TDX server failed."""


class TDXLoginError(TDXError):
    """Login to trading account failed."""


class TDXTradeError(TDXError):
    """Trading operation failed."""


class TDXSessionExpiredError(TDXError):
    """Session has expired, need to re-login."""
