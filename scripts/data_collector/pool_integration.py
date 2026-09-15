"""Integration boundary between market-data collectors and dynamic pools.

Collectors own downloading, normalization, binary dumping, and verification.
This module owns the optional post-success refresh of the personal daily pool,
so source-specific collectors do not need to know the pool implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def update_personal_pool_after_data_update(
    *,
    provider_uri: Union[str, Path],
    region: str,
    storage_freq: str,
    enabled: bool = False,
    update_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_name: str = "personal_dynamic_pool",
):
    """Refresh the personal pool after a successful daily data update.

    The function is intentionally a no-op unless ``enabled`` is true.  A
    caller may provide one exact date or an inclusive range; with neither,
    the latest trading date in the provider calendar is refreshed.
    """

    if not enabled:
        return None
    if str(region).lower() != "cn":
        raise ValueError("personal pool update is only supported for cn region data")
    if storage_freq != "day":
        raise ValueError("personal pool update is only supported for daily data")
    if update_date is not None and (start_date is not None or end_date is not None):
        raise ValueError("Specify either update_date or start_date/end_date, not both")
    if (start_date is None) != (end_date is None):
        raise ValueError("personal pool range updates require both start_date and end_date")

    from quant_master.data.personal_dynamic_pool import (  # pylint: disable=import-outside-toplevel
        update_latest_personal_dynamic_pool,
        update_personal_dynamic_pool,
    )

    pool_kwargs = {"provider_uri": Path(provider_uri), "output_name": output_name}
    if update_date is not None or start_date is not None or end_date is not None:
        result = update_personal_dynamic_pool(
            **pool_kwargs,
            update_date=update_date,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        result = update_latest_personal_dynamic_pool(**pool_kwargs)
    return result


__all__ = ["update_personal_pool_after_data_update"]
