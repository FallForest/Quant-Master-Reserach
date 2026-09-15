"""Atomic status manifest shared by market-data update entrypoints."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union


def write_update_manifest(
    provider_uri: Union[str, Path],
    *,
    source: str,
    interval: str,
    storage_freq: str,
    start_date: str,
    end_date: str,
    stages: Mapping[str, str],
    options: Mapping[str, Any],
) -> Dict[str, Any]:
    """Write and return the latest update status using an atomic replace."""

    provider_dir = Path(provider_uri).expanduser().resolve()
    provider_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": source,
        "interval": interval,
        "storage_freq": storage_freq,
        "start_date": start_date,
        "end_date": end_date,
        "stages": dict(stages),
        "options": dict(options),
    }
    manifest_path = provider_dir / "update_manifest.json"
    temp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    temp_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(manifest_path)
    return manifest


__all__ = ["write_update_manifest"]
