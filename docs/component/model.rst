.. _model:

===========================================
Forecast Model: Model Training & Prediction
===========================================

Introduction
============

``Forecast Model`` is designed to make the `prediction score` about stocks. Users can use the ``Forecast Model`` in an automatic workflow by ``qrun``, please refer to `Workflow: Workflow Management <workflow.html>`_.

Because the components in ``QuantMaster`` are designed in a loosely-coupled way, ``Forecast Model`` can be used as an independent module also.

Base Class & Interface
======================

``QuantMaster`` provides a base class `quant_master.model.base.Model <../reference/api.html#module-quant_master.model.base>`_ from which all models should inherit.

The base class provides the following interfaces:

.. autoclass:: quant_master.model.base.Model
    :members:
    :noindex:

``QuantMaster`` also provides a base class `quant_master.model.base.ModelFT <../reference/api.html#quant_master.model.base.ModelFT>`_, which includes the method for finetuning the model.

For other interfaces such as `finetune`, please refer to `Model API <../reference/api.html#module-quant_master.model.base>`_.

Example
=======

``QuantMaster``'s `Model Zoo` includes models such as ``LightGBM``, ``MLP``, ``LSTM``, etc.. These models are treated as the baselines of ``Forecast Model``. The following steps show how to run`` LightGBM`` as an independent module.

- Initialize ``QuantMaster`` with `quant_master.init` first, please refer to `Initialization <../start/initialization.html>`_.
- Run the following code to get the `prediction score` `pred_score`
    .. code-block:: Python

        from quant_master.contrib.model.gbdt import LGBModel
        from quant_master.contrib.data.handler import Alpha158
        from quant_master.utils import init_instance_by_config, flatten_dict
        from quant_master.workflow import R
        from quant_master.workflow.record_temp import SignalRecord, PortAnaRecord

        market = "csi300"
        benchmark = "SH000300"

        data_handler_config = {
            "start_time": "2008-01-01",
            "end_time": "2020-08-01",
            "fit_start_time": "2008-01-01",
            "fit_end_time": "2014-12-31",
            "instruments": market,
        }

        task = {
            "model": {
                "class": "LGBModel",
                "module_path": "quant_master.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.0421,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                },
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "quant_master.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "quant_master.contrib.data.handler",
                        "kwargs": data_handler_config,
                    },
                    "segments": {
                        "train": ("2008-01-01", "2014-12-31"),
                        "valid": ("2015-01-01", "2016-12-31"),
                        "test": ("2017-01-01", "2020-08-01"),
                    },
                },
            },
        }

        # model initialization
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])

        # start exp
        with R.start(experiment_name="workflow"):
            # train
            R.log_params(**flatten_dict(task))
            model.fit(dataset)

            # prediction
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

    .. note::

        `Alpha158` is the data handler provided by ``QuantMaster``, please refer to `Data Handler <data.html#data-handler>`_.
        `SignalRecord` is the `Record Template` in ``QuantMaster``, please refer to `Workflow <recorder.html#record-template>`_.

Also, the above example has been given in ``examples/train_backtest_analyze.ipynb``.
Technically, the meaning of the model prediction depends on the label setting designed by user.
By default, the meaning of the score is normally the rating of the instruments by the forecasting model. The higher the score, the more profit the instruments.


Custom Model
============

QuantMaster supports custom models. If users are interested in customizing their own models and integrating the models into ``QuantMaster``, please refer to `Custom Model Integration <../start/integration.html>`_.


API
===
Please refer to `Model API <../reference/api.html#module-quant_master.model.base>`_.
