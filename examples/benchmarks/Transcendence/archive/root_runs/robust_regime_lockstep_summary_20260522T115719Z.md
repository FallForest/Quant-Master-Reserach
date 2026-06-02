# Robust Regime Lockstep Summary (20260522T115719Z)

- Protocol: lockstep / walk-forward, select on prior window only.
- Test period: `2024-01-01..2026-04-30`.
- Costs: `open=0.0005`, `close=0.0015`.
- Hard gate: `IR>2.9` and `AnnRet>0.27`.
- Hard gate pass: `False`.
- Stitched metrics: `{"annret": 0.02031034054187492, "ir": 0.17669234825688596, "max_drawdown": -0.05964488739493712}`.
- Degrade notes: `["selection window 2022_2023 unavailable (pred starts 2024-01-02); degrade to 2024H1->2024H2.", "selection window 2024 clipped start 2024-01-01 -> 2024-01-02 due to pred coverage.", "selection window up_to_2025 clipped start 2024-01-01 -> 2024-01-02 due to pred coverage."]`.
