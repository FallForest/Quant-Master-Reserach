# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import torch

from quant_master.contrib.model.double_ensemble import DEnsembleModel
from quant_master.contrib.model.pytorch_utils import deepcopy_state_dict
from quant_master.contrib.model.transcendence_objective import build_objective_label_frame


def test_deepcopy_state_dict_is_not_mutated_by_later_training_updates():
    model = torch.nn.Linear(2, 1)
    snapshot = deepcopy_state_dict(model)

    with torch.no_grad():
        model.weight.add_(1.0)
        model.bias.add_(1.0)

    assert not torch.equal(snapshot["weight"], model.state_dict()["weight"])
    assert not torch.equal(snapshot["bias"], model.state_dict()["bias"])


def test_double_ensemble_propagates_random_state_to_lightgbm():
    model = DEnsembleModel(random_state=17, subsample=0.8)

    assert model.params["seed"] == 17
    assert model.params["feature_fraction_seed"] == 17
    assert model.params["bagging_seed"] == 17
    assert model.params["data_random_seed"] == 17
    assert model.params["bagging_freq"] == 1


def test_double_ensemble_preserves_explicit_lightgbm_seed_and_bagging_frequency():
    model = DEnsembleModel(random_state=17, seed=23, bagging_fraction=0.7, bagging_freq=4)

    assert model.params["seed"] == 23
    assert model.params["bagging_freq"] == 4


def test_objective_label_does_not_use_partial_forward_windows_at_segment_tail():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2022-01-03", periods=6), ["a"]], names=["datetime", "instrument"]
    )
    labels = pd.DataFrame({"label": np.arange(6, dtype=float)}, index=index)

    objective = build_objective_label_frame(labels, base_horizon=2)

    np.testing.assert_allclose(objective.iloc[:3, 0].to_numpy(), [0.5, 1.5, 2.5])
    assert objective.iloc[3:, 0].isna().all()
