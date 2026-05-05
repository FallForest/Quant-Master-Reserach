# MetaEnsemble
* MetaEnsemble combines three tabular alpha models that are already native to Qlib: DoubleEnsemble, LightGBM, and Linear.
* The model trains each sub-model on the same dataset and learns blend weights from validation performance.
* It is designed as a higher-conviction benchmark candidate specifically for Alpha158-style cross-sectional stock selection.
