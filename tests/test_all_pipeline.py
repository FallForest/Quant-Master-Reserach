# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import shutil
import unittest
import pytest
import tempfile
from pathlib import Path

import pandas as pd

import quant_master
from quant_master.config import C
from quant_master.data import D
from quant_master.data.filter import NameDFilter
from quant_master.tests import TestAutoData
from quant_master.tests.config import CSI300_GBDT_TASK, CSI300_BENCH
from quant_master.utils import init_instance_by_config, flatten_dict
from quant_master.workflow import R
from quant_master.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord


def train(uri_path: str = None):
    """train model

    Returns
    -------
        pred_score: pandas.DataFrame
            predict scores
        performance: dict
            model performance
    """

    # model initialization
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    # To test __repr__
    print(dataset)
    print(R)

    # start exp
    with R.start(experiment_name="workflow", uri=uri_path):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        # prediction
        recorder = R.get_recorder()
        # To test __repr__
        print(recorder)
        # To test get_local_dir
        print(recorder.get_local_dir())
        rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        pred_score = sr.load("pred.pkl")

        # calculate ic and ric
        sar = SigAnaRecord(recorder)
        sar.generate()
        ic = sar.load("ic.pkl")
        ric = sar.load("ric.pkl")

        uri_path = R.get_uri()
    return pred_score, {"ic": ic, "ric": ric}, rid, uri_path


def fake_experiment():
    """A fake experiment workflow to test uri

    Returns
    -------
        pass_or_not_for_default_uri: bool
        pass_or_not_for_current_uri: bool
        temporary_exp_dir: str
    """

    # start exp
    default_uri = R.get_uri()
    current_uri = "file:./temp-test-exp-mag"
    with R.start(experiment_name="fake_workflow_for_expm", uri=current_uri):
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))

        current_uri_to_check = R.get_uri()
    default_uri_to_check = R.get_uri()
    return default_uri == default_uri_to_check, current_uri == current_uri_to_check, current_uri


def build_manual_prediction():
    instruments = D.instruments("csi300", filter_pipe=[NameDFilter(name_rule_re="SH600110")])
    pred = D.features(instruments, ["$close"], start_time="2005-01-04", end_time="2005-01-14")
    pred = pred.rename(columns={"$close": "score"})
    return pred


def backtest_analysis(pred, rid, uri_path: str = None):
    """backtest and analysis

    Parameters
    ----------
    rid : str
        the id of the recorder to be used in this function
    uri_path: str
        mlflow uri path

    Returns
    -------
    analysis : pandas.DataFrame
        the analysis result

    """
    with R.uri_context(uri=uri_path):
        recorder = R.get_recorder(experiment_name="workflow", recorder_id=rid)

    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    model = recorder.load_object("trained_model")

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
    # backtest
    par = PortAnaRecord(recorder, port_analysis_config, risk_analysis_freq="day")
    par.generate()
    analysis_df = par.load("port_analysis_1day.pkl")
    print(analysis_df)
    return analysis_df


def backtest_analysis_with_router(recorder, pred):
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
            "class": "DailyRebalanceRouterStrategy",
            "module_path": "quant_master.contrib.strategy",
            "kwargs": {
                "selector": {
                    "type": "series",
                    "signal": pd.Series(
                        [
                            {"family": "topk", "variant": "aggressive", "reason": "trend_following"},
                            {"family": "buffered", "variant": "base", "reason": "cost_defense"},
                        ],
                        index=pd.to_datetime(["2005-01-04", "2005-01-05"]),
                    ),
                },
                "default_family": "topk",
                "default_variant": "aggressive",
                "strategy_families": {
                    "topk": {
                        "default_variant": "aggressive",
                        "variants": {
                            "aggressive": {
                                "class": "TopkDropoutStrategy",
                                "module_path": "quant_master.contrib.strategy.signal_strategy",
                                "kwargs": {
                                    "signal": "<PRED>",
                                    "topk": 1,
                                    "n_drop": 1,
                                },
                            }
                        },
                    },
                    "buffered": {
                        "default_variant": "base",
                        "variants": {
                            "base": {
                                "class": "TopkDropoutStrategy",
                                "module_path": "quant_master.contrib.strategy.signal_strategy",
                                "kwargs": {
                                    "signal": "<PRED>",
                                    "topk": 1,
                                    "n_drop": 0,
                                },
                            }
                        },
                    },
                },
            },
        },
        "backtest": {
            "start_time": "2005-01-04",
            "end_time": "2005-01-13",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0001,
                "close_cost": 0.0006,
                "min_cost": 0,
                "codes": "csi300",
            },
        },
    }
    recorder.save_objects(**{"pred.pkl": pred, "label.pkl": pred[["score"]].rename(columns={"score": "label"})})
    par = PortAnaRecord(recorder, port_analysis_config, risk_analysis_freq=[])
    artifacts = par.generate()
    report_df = par.load("report_normal_1day.pkl")
    route_df = par.load("strategy_route_1day.pkl")
    route_summary_df = par.load("strategy_route_summary_1day.pkl")
    return report_df, route_df, route_summary_df, artifacts


class TestAllFlow(TestAutoData):
    REPORT_NORMAL = None
    POSITIONS = None
    RID = None
    URI_PATH = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._temp_dir = tempfile.TemporaryDirectory(prefix="quant-master-test-all-flow-")
        cls.URI_PATH = "file:" + str(Path(cls._temp_dir.name).resolve())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    @pytest.mark.slow
    def test_0_train(self):
        TestAllFlow.PRED_SCORE, ic_ric, TestAllFlow.RID, uri_path = train(self.URI_PATH)
        self.assertGreaterEqual(ic_ric["ic"].all(), 0, "train failed")
        self.assertGreaterEqual(ic_ric["ric"].all(), 0, "train failed")

    @pytest.mark.slow
    def test_1_backtest(self):
        analyze_df = backtest_analysis(TestAllFlow.PRED_SCORE, TestAllFlow.RID, self.URI_PATH)
        self.assertGreaterEqual(
            analyze_df.loc(axis=0)["excess_return_with_cost", "annualized_return"].values[0],
            0.05,
            "backtest failed",
        )
        self.assertTrue(not analyze_df.isna().any().any(), "backtest failed")

    @pytest.mark.slow
    def test_2_router_backtest(self):
        with R.start(experiment_name="workflow_router", uri=self.URI_PATH):
            recorder = R.get_recorder()
            pred = build_manual_prediction()
            report_df, route_df, route_summary_df, artifacts = backtest_analysis_with_router(recorder, pred)
        self.assertFalse(report_df.empty, "router backtest report should not be empty")
        self.assertFalse(route_df.empty, "router route history should not be empty")
        self.assertIn("strategy_key", route_df.columns, "route history should include selected strategy key")
        self.assertIn("strategy_family", route_df.columns, "route history should include strategy family")
        self.assertIn("strategy_variant", route_df.columns, "route history should include strategy variant")
        self.assertIn("reason", route_df.columns, "route history should include selection reason")
        self.assertFalse(route_summary_df.empty, "router route summary should not be empty")
        self.assertIn("selection_count", route_summary_df.columns, "route summary should include selection count")
        self.assertIn("strategy_route_1day.pkl", artifacts, "router route artifact should be generated")
        self.assertIn(
            "strategy_route_summary_1day.pkl", artifacts, "router route summary artifact should be generated"
        )

    @pytest.mark.slow
    def test_3_expmanager(self):
        pass_default, pass_current, uri_path = fake_experiment()
        self.assertTrue(pass_default, msg="default uri is incorrect")
        self.assertTrue(pass_current, msg="current uri is incorrect")
        shutil.rmtree(str(Path(uri_path.strip("file:")).resolve()))


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_train"))
    _suite.addTest(TestAllFlow("test_1_backtest"))
    _suite.addTest(TestAllFlow("test_2_router_backtest"))
    _suite.addTest(TestAllFlow("test_3_expmanager"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
