# TreeCnLstmRL

TreeCnLstmRL is a two-model alpha ensemble with a validation-driven RL bandit blender.

* The tree branch uses LightGBM on cross-sectional Alpha158 features.
* The CnLSTM branch uses a CNN front-end followed by LSTM over `TSDatasetH` windows.
* The RL blender learns final model weights from validation rewards, using daily rank IC by default.

The workflow uses one `TSDatasetH` config. The tree branch reads tabular data directly from the dataset handler, while
CnLSTM consumes the time-series sampler.
