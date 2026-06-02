# Lockstep After Epsilon Recheck (20260522T151307Z)

- Strict baseline before fix artifact: `{"annret": 0.02031034054187492, "ir": 0.17669234825688596, "max_drawdown": -0.05964488739493712}`
- Strict baseline after fix: `{"annret": 0.2518930639300241, "ir": 1.8754870651622841, "max_drawdown": -0.04439603009175652}`
- Diagnostic safe wrapper: `{"annret": 0.25189304157549136, "ir": 1.875486897720126, "max_drawdown": -0.044396031114038476}`

## Gate Distance
- strict after fix IR shortfall: `1.0245129348377158`; AnnRet shortfall: `0.018106936069975932`
- safe wrapper IR shortfall: `1.024513102279874`; AnnRet shortfall: `0.018106958424508657`

## Attribution
- Real bug fix: production `Position._sell_stock` now clips tiny floating-point oversell and restores the 2025 strict replay instead of throwing.
- Wrapper only: `min_names_guard`, dynamic `topk` clamp, cash fallback, and safe order-generator clipping alter diagnostic runtime behavior and are not production lockstep results.

## Conclusion
- Epsilon fix alone is not enough to get close to the hard gate.
- Any apparent near-recovery that depends on the safe wrapper should be treated as diagnostic evidence, not a new approved model result.
