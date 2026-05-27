# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""Low-level TDX HTTP client implementing the sessionTs protocol.

The TDX web trading UI communicates via:
    GET  /TOUCH?Device=Browser&Ip=...&Mac=...&Build=WEB&Type=41&Ver=1.0.0&EP=0
    POST /TQL?Entry=<function_name>&RI=<random_id>
    POST /TQLEX?Entry=<function_name>           (extended, no RI)
    GET  /ALIVE?
    GET  /QUIT?

The /TQLEX endpoint is confirmed from the web JS source (sessionTX/sessionTs classes).
The /TQL endpoint adds an RI (resource identifier) parameter from localStorage.
"""

import random
import time
from typing import Dict, Optional
from urllib.parse import urlencode

import requests

from quant_master.log import get_module_logger

from .consts import (
    COOKIE_TDXID,
    COOKIE_TOKEN,
    SESSION_TIMEOUT_SEC,
    TOUCH_PARAMS,
    PATH_TOUCH,
    PATH_TQL,
    PATH_TQLEX,
    PATH_ALIVE,
    PATH_QUIT,
)
from .exceptions import TDXConnectionError, TDXTradeError

logger = get_module_logger("TDXClient")


class TDXClient:
    """Low-level HTTP client for TDX trading protocol."""

    DEFAULT_TIMEOUT = 30

    def __init__(
        self,
        host: str,
        port: int = 7708,
        use_https: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.host = host
        self.port = port
        self.scheme = "https" if use_https else "http"
        self.base_url = f"{self.scheme}://{host}:{port}"
        self.timeout = timeout
        self._session: Optional[requests.Session] = None
        self._last_activity: float = 0

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _random_ri(self) -> str:
        return f"{random.randint(0, 0xFFFFFFFF):08X}"

    @property
    def cookies(self) -> dict:
        if self._session:
            return dict(self._session.cookies)
        return {}

    def connect(self) -> bool:
        """Perform TOUCH handshake and establish HTTP session."""
        self._session = requests.Session()
        try:
            resp = self._session.get(
                self._url(PATH_TOUCH),
                params=TOUCH_PARAMS,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self._last_activity = time.time()

            tdxid = self._session.cookies.get(COOKIE_TDXID)
            if tdxid:
                logger.info(f"Connected to {self.base_url}, TDXID={tdxid}")
                return True

            # Some servers return OK but set cookies differently
            logger.warning("TOUCH OK but no TDXID cookie; checking response body")
            return bool(resp.text.strip())

        except requests.RequestException as e:
            self._session = None
            raise TDXConnectionError(f"Cannot connect to {self.base_url}: {e}")

    def _ensure_session(self):
        """Check session freshness, send heartbeat if near expiry."""
        if self._session is None:
            raise TDXConnectionError("Not connected. Call connect() first.")
        if time.time() - self._last_activity > SESSION_TIMEOUT_SEC - 60:
            logger.info("Session near expiry, sending ALIVE")
            self.alive()

    def call_tql(
        self,
        entry: str,
        data: Optional[Dict[str, str]] = None,
    ) -> str:
        """Make a TQL API call via /TQL endpoint (with RI parameter).

        Parameters
        ----------
        entry : str
            TQL Entry function name, e.g. "ACL.checkuser"
        data : dict, optional
            Form-encoded POST body parameters.

        Returns
        -------
        str
            Response body text.
        """
        self._ensure_session()

        ri = self._random_ri()
        url = self._url(f"{PATH_TQL}?Entry={entry}&RI={ri}")

        try:
            if data:
                resp = self._session.post(
                    url,
                    data=urlencode(data),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=self.timeout,
                )
            else:
                resp = self._session.get(url, timeout=self.timeout)

            self._last_activity = time.time()
            resp.raise_for_status()
            return resp.text

        except requests.RequestException as e:
            logger.error(f"TQL call '{entry}' failed: {e}")
            raise TDXTradeError(f"TQL call '{entry}' failed: {e}")

    def call_tqlex(
        self,
        entry: str,
        data: Optional[Dict[str, str]] = None,
    ) -> str:
        """Make a TQL API call via /TQLEX endpoint (no RI parameter).

        Confirmed from web JS source: sessionTX/sessionTs.send() uses
        url: "/TQLEX?Entry=" + t, type: "POST", data: e

        Parameters
        ----------
        entry : str
            TQL Entry function name, e.g. "Stock.Buy"
        data : dict, optional
            Form-encoded POST body parameters.

        Returns
        -------
        str
            Response body text.
        """
        self._ensure_session()

        url = self._url(f"{PATH_TQLEX}?Entry={entry}")

        try:
            if data:
                resp = self._session.post(
                    url,
                    data=urlencode(data),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=self.timeout,
                )
            else:
                resp = self._session.get(url, timeout=self.timeout)

            self._last_activity = time.time()
            resp.raise_for_status()
            return resp.text

        except requests.RequestException as e:
            logger.error(f"TQLEX call '{entry}' failed: {e}")
            raise TDXTradeError(f"TQLEX call '{entry}' failed: {e}")

    def call(self, entry: str, data: Optional[Dict[str, str]] = None, extended: bool = True) -> str:
        """Unified TQL call. Uses /TQLEX by default, /TQL if extended=False."""
        if extended:
            return self.call_tqlex(entry, data)
        return self.call_tql(entry, data)

    def alive(self) -> bool:
        """Send heartbeat to keep session alive."""
        if self._session is None:
            return False
        try:
            resp = self._session.get(self._url(PATH_ALIVE), timeout=self.timeout)
            self._last_activity = time.time()
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def quit(self) -> None:
        """End the session."""
        if self._session:
            try:
                self._session.get(self._url(PATH_QUIT), timeout=self.timeout)
            except requests.RequestException:
                pass
            self._session = None
            logger.info("Disconnected")

    def is_connected(self) -> bool:
        return self._session is not None and self.alive()
