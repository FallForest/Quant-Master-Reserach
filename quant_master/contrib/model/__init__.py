# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
try:
    from .catboost_model import CatBoostModel
except ModuleNotFoundError:
    CatBoostModel = None
    print("ModuleNotFoundError. CatBoostModel are skipped. (optional: maybe installing CatBoostModel can fix it.)")
try:
    from .double_ensemble import DEnsembleModel
    from .gbdt import LGBModel
except ModuleNotFoundError:
    DEnsembleModel, LGBModel = None, None
    print(
        "ModuleNotFoundError. DEnsembleModel and LGBModel are skipped. (optional: maybe installing lightgbm can fix it.)"
    )
try:
    from .adaptive_ensemble import AdaptiveEnsembleModel
except ModuleNotFoundError:
    AdaptiveEnsembleModel = None
    print(
        "ModuleNotFoundError. AdaptiveEnsembleModel is skipped. (optional: maybe installing sklearn can fix it.)"
    )
try:
    from .meta_ensemble import MetaEnsembleModel
except ModuleNotFoundError:
    MetaEnsembleModel = None
    print(
        "ModuleNotFoundError. MetaEnsembleModel is skipped. (optional: maybe installing sklearn can fix it.)"
    )
try:
    from .dynamic_meta_ensemble import DynamicMetaEnsembleModel
except ModuleNotFoundError:
    DynamicMetaEnsembleModel = None
    print(
        "ModuleNotFoundError. DynamicMetaEnsembleModel is skipped. "
        "(optional: maybe installing lightgbm and sklearn can fix it.)"
    )
try:
    from .low_turnover_double_ensemble import LowTurnoverDEnsembleModel
except ModuleNotFoundError:
    LowTurnoverDEnsembleModel = None
    print("ModuleNotFoundError. LowTurnoverDEnsembleModel is skipped. (optional: maybe installing lightgbm can fix it.)")
try:
    from .residual_double_ensemble_lgb import ResidualDEnsembleLGBModel
except ModuleNotFoundError:
    ResidualDEnsembleLGBModel = None
    print("ModuleNotFoundError. ResidualDEnsembleLGBModel is skipped. (optional: maybe installing lightgbm can fix it.)")
try:
    from .multiseed_double_ensemble import MultiSeedDEnsembleModel
except ModuleNotFoundError:
    MultiSeedDEnsembleModel = None
    print("ModuleNotFoundError. MultiSeedDEnsembleModel is skipped. (optional: maybe installing lightgbm can fix it.)")
try:
    from .cost_aware_double_ensemble import CostAwareDEnsembleModel
except ModuleNotFoundError:
    CostAwareDEnsembleModel = None
    print("ModuleNotFoundError. CostAwareDEnsembleModel is skipped. (optional: maybe installing lightgbm can fix it.)")
try:
    from .tree_cn_lstm_rl import TreeCnLstmRLModel
    from .double_ensemble_residual_cn_lstm import DoubleEnsembleResidualCnLstmModel
except ModuleNotFoundError:
    TreeCnLstmRLModel, DoubleEnsembleResidualCnLstmModel = None, None
    print(
        "ModuleNotFoundError. TreeCnLstmRLModel and DoubleEnsembleResidualCnLstmModel are skipped. "
        "(optional: maybe installing lightgbm and pytorch can fix it.)"
    )
try:
    from .pretrained_signal import PretrainedSignalModel
except ModuleNotFoundError:
    PretrainedSignalModel = None
    print("ModuleNotFoundError. PretrainedSignalModel is skipped.")
try:
    from .xgboost import XGBModel
except ModuleNotFoundError:
    XGBModel = None
    print("ModuleNotFoundError. XGBModel is skipped(optional: maybe installing xgboost can fix it).")
try:
    from .linear import LinearModel
except ModuleNotFoundError:
    LinearModel = None
    print("ModuleNotFoundError. LinearModel is skipped(optional: maybe installing scipy and sklearn can fix it).")
# import pytorch models
try:
    from .pytorch_alstm import ALSTM
    from .pytorch_gats import GATs
    from .pytorch_gru import GRU
    from .pytorch_lstm import LSTM
    from .pytorch_nn import DNNModelPytorch
    from .pytorch_tabnet import TabnetModel
    from .pytorch_sfm import SFM_Model
    from .pytorch_tcn import TCN
    from .pytorch_add import ADD
    from .pytorch_tft import TFTModel
    from .pytorch_adarnn import ADARNN
    from .pytorch_localformer import LocalformerModel
    from .pytorch_hist import HIST
    from .pytorch_krnn import KRNN
    from .pytorch_igmtf import IGMTF
    from .pytorch_sandwich import Sandwich
    from .pytorch_tcts import TCTS
    from .pytorch_tra import TRAModel
    from .pytorch_general_nn import GeneralPTNN

    pytorch_classes = (
        ALSTM, GATs, GRU, LSTM, DNNModelPytorch, TabnetModel, SFM_Model, TCN, ADD, TFTModel,
        ADARNN, LocalformerModel, HIST, KRNN, IGMTF, Sandwich, TCTS, TRAModel, GeneralPTNN,
    )
except ModuleNotFoundError:
    pytorch_classes = ()
    print("ModuleNotFoundError.  PyTorch models are skipped (optional: maybe installing pytorch can fix it).")
try:
    from .regime_horizon_cost_ensemble import RegimeHorizonCostEnsembleModel
    from .transcendence_hybrid import TranscendenceHybridModel
    from .transcendence_signal_ensemble import TranscendenceSignalEnsembleModel
except ModuleNotFoundError:
    RegimeHorizonCostEnsembleModel = None
    TranscendenceHybridModel = None
    TranscendenceSignalEnsembleModel = None
    print("ModuleNotFoundError. Transcendence models are skipped.")
try:
    from .topk_meta_label_model import TopKMetaLabelModel
except ModuleNotFoundError:
    TopKMetaLabelModel = None
    print("ModuleNotFoundError. TopKMetaLabelModel is skipped. (optional: maybe installing lightgbm can fix it.)")
try:
    from .highfreq_gdbt_model import HFLGBModel
except ModuleNotFoundError:
    HFLGBModel = None
    print("ModuleNotFoundError. HFLGBModel is skipped. (optional: maybe installing lightgbm can fix it.)")

all_model_classes = (
    CatBoostModel,
    DEnsembleModel,
    LGBModel,
    AdaptiveEnsembleModel,
    MetaEnsembleModel,
    DynamicMetaEnsembleModel,
    LowTurnoverDEnsembleModel,
    ResidualDEnsembleLGBModel,
    MultiSeedDEnsembleModel,
    CostAwareDEnsembleModel,
    TreeCnLstmRLModel,
    DoubleEnsembleResidualCnLstmModel,
    PretrainedSignalModel,
    XGBModel,
    LinearModel,
    RegimeHorizonCostEnsembleModel,
    TranscendenceHybridModel,
    TranscendenceSignalEnsembleModel,
    TopKMetaLabelModel,
    HFLGBModel,
) + pytorch_classes
