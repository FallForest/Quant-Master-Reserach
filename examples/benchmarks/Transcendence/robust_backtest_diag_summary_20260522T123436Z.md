# Robust Backtest Constraint Diagnostics (20260522T123436Z)

- Window: `2025-01-01..2025-12-31`
- Failed candidate count: `2`
- Failed events parsed: `3`
- Baseline stitched metrics (from lockstep summary): `{"annret": 0.02031034054187492, "ir": 0.17669234825688596, "max_drawdown": -0.05964488739493712}`
- Safe stitched metrics (diagnostic wrapper): `{"annret": 0.02031031722690878, "ir": 0.17669214565114122, "max_drawdown": -0.05964488819116066}`

## Root Cause Snapshot
- candidate `7735252674316944203`: min selected `35`, min tradable selected `22`, days selected<=1 `0`, days tradable selected<topk `198`; baseline error `{'error': 'only have 5161.309634309716 SH603296, require 5161.39746182772', 'only_have': 5161.309634309716, 'symbol': 'SH603296', 'require': 5161.39746182772, 'diff': 0.08782751800390542, 'order_stock_id': 'SH603296', 'order_direction': 0, 'order_amount': nan, 'trade_val': 6722.000358019846, 'trade_price': 1.3023605346679688, 'trade_start_time': '2025-01-13 00:00:00', 'trade_end_time': '2025-01-13 23:59:59', 'before_amount': 5161.309634309716, 'nonnull_ratio': 0.1281207133058985, 'stress_ratio': 0.31275720164609055, 'topk_effective_max': 40, 'safe_mode': False}`

## Safe Fallback (Diagnostic Only)
- min_names guard + dynamic topk clamp + cash fallback + sell epsilon clip.
- Implemented only in this script wrapper, not in production strategy library.
