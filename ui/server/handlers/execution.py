"""Execution HTTP handlers."""

from __future__ import annotations

from ..execution_service import build_order_preview, get_execution_config, load_history, submit_orders


def config(rh):
    rh._json_response(get_execution_config())


def preview(rh):
    try:
        body = rh._read_body()
    except Exception:
        return

    trades = body.get("trades") or []
    risk = body.get("risk") or {}
    payload = build_order_preview(trades, risk=risk)
    rh._json_response(payload)


def submit(rh):
    try:
        body = rh._read_body()
    except Exception:
        return

    try:
        payload = submit_orders(
            body.get("orders") or [],
            broker_kind=body.get("brokerKind"),
            dry_run=body.get("dryRun"),
            risk=body.get("risk") or {},
            confirm=bool(body.get("confirm")),
        )
    except ValueError as exc:
        return rh._json_response({"error": str(exc)}, status=400)
    except PermissionError as exc:
        return rh._json_response({"error": str(exc)}, status=403)
    except Exception as exc:
        return rh._json_response({"error": str(exc)}, status=500)

    rh._json_response(payload)


def history(rh):
    params = rh._query_params()
    limit = int(params.get("limit", 30))
    rh._json_response(load_history(limit=limit))
