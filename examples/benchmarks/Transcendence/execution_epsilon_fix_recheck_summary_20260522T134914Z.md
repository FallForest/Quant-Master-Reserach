# Robust Backtest Constraint Diagnostics (20260522T134914Z)

- Window: `2025-01-13..2025-01-13`
- Failed candidate count: `2`
- Failed events parsed: `3`
- Baseline stitched metrics (from lockstep summary): `{"annret": 0.02031034054187492, "ir": 0.17669234825688596, "max_drawdown": -0.05964488739493712}`
- Safe stitched metrics (diagnostic wrapper): `{"annret": 0.2518930415754917, "ir": 1.8754868977201284, "max_drawdown": -0.04439603111403806}`

## Root Cause Snapshot
- candidate `7735252674316944203`: min selected `40`, min tradable selected `37`, days selected<=1 `0`, days tradable selected<topk `1`; baseline error `None`

## Safe Fallback (Diagnostic Only)
- min_names guard + dynamic topk clamp + cash fallback + sell epsilon clip.
- Implemented only in this script wrapper, not in production strategy library.
