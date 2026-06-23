#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
QuantMaster provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) QuantMaster is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""

import quant_master
from quant_master.constant import REG_CN
from quant_master.utils import init_instance_by_config, flatten_dict
from quant_master.workflow import R
from quant_master.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from quant_master.tests.data import GetData
from quant_master.tests.config import CSI300_BENCH, CSI300_GBDT_TASK

if __name__ == "__main__":
    # use default data
    provider_uri = "~/.quant_master/quant_master_data/cn_data"  # target_dir
    GetData().quant_master_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    quant_master.init(provider_uri=provider_uri, region=REG_CN)

    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "quant_master.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "quant_master.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0001,
                "close_cost": 0.0006,
                "min_cost": 0,
            },
        },
    }

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    example_df = dataset.prepare("train")
    print(example_df.head())

    # start exp
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://quant_master.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
