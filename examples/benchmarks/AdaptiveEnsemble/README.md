# AdaptiveEnsemble
* AdaptiveEnsemble is a heterogeneous tabular ensemble for Alpha158-style cross-sectional prediction.
* It trains LightGBM, ExtraTrees, and Ridge sub-models on the same feature set, then learns validation-based ensemble weights from cross-sectional standardized predictions.
* The implementation is designed to stay within QuantMaster's native benchmark workflow and dependency stack.
