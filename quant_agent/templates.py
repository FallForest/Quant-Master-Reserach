"""Inlined Quant-Master workflow YAML templates previously loaded from vendor/RD-Agent."""

FACTOR_WORKFLOW_TEMPLATE = """\
quant_master_init:
    provider_uri: "~/.quant_master/quant_master_data/cn_data"
    region: cn

market: &market csi300
benchmark: &benchmark SH000300

data_handler_config: &data_handler_config
    start_time: {{ train_start | default("2008-01-01", true) }}
    end_time: {{ test_end | default("null", true) }}
    instruments: *market
    data_loader:
        class: NestedDataLoader
        kwargs:
            dataloader_l:
                - class: quant_master.contrib.data.loader.Alpha158DL
                  kwargs:
                    config:
                        label:
                            - ["Ref($close, -2)/Ref($close, -1) - 1"]
                            - ["LABEL0"]
                        feature:
                            - {{ feature_expressions }}
                            - {{ feature_names }}

    infer_processors:
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
              fit_start_time: {{ train_start | default("2008-01-01", true) }}
              fit_end_time: {{ train_end | default("2014-12-31", true) }}
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSZScoreNorm
          kwargs:
              fields_group: label

port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: quant_master.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5
    backtest:
        start_time: {{ test_start | default("2017-01-01", true) }}
        end_time: {{ test_end | default("null", true) }}
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5

task:
    model:
        class: LGBModel
        module_path: quant_master.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.2
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 20
    dataset:
        class: DatasetH
        module_path: quant_master.data.dataset
        kwargs:
            handler:
                class: DataHandlerLP
                module_path: quant_master.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [{{ train_start | default("2008-01-01", true) }}, {{ train_end | default("2014-12-31", true) }}]
                valid: [{{ valid_start | default("2015-01-01", true) }}, {{ valid_end | default("2016-12-31", true) }}]
                test: [{{ test_start | default("2017-01-01", true) }}, {{ test_end | default("null", true) }}]
    record:
        - class: SignalRecord
          module_path: quant_master.workflow.record_temp
          kwargs:
            model: <MODEL>
            dataset: <DATASET>
        - class: SigAnaRecord
          module_path: quant_master.workflow.record_temp
          kwargs:
            ana_long_short: False
            ann_scaler: 252
        - class: PortAnaRecord
          module_path: quant_master.workflow.record_temp
          kwargs:
            config: *port_analysis_config
"""

MODEL_WORKFLOW_TEMPLATE = """\
quant_master_init:
    provider_uri: "~/.quant_master/quant_master_data/cn_data"
    region: cn
market: &market csi300
benchmark: &benchmark SH000300

data_handler_config: &data_handler_config
    start_time: {{ train_start | default("2008-01-01", true) }}
    end_time: {{ test_end | default("null", true) }}
    instruments: *market
    data_loader:
        class: NestedDataLoader
        kwargs:
            dataloader_l:
                - class: quant_master.contrib.data.loader.Alpha158DL
                  kwargs:
                    config:
                        label:
                            - ["Ref($close, -2)/Ref($close, -1) - 1"]
                            - ["LABEL0"]
                        feature:
                            - {{ feature_expressions }}
                            - {{ feature_names }}

    infer_processors:
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
              fit_start_time: {{ train_start | default("2008-01-01", true) }}
              fit_end_time: {{ train_end | default("2014-12-31", true) }}
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSZScoreNorm
          kwargs:
              fields_group: label

port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: quant_master.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5
    backtest:
        start_time: {{ test_start | default("2017-01-01", true) }}
        end_time: {{ test_end | default("null", true) }}
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5
task:
    model:
        class: GeneralPTNN
        module_path: quant_master.contrib.model.pytorch_general_nn
        kwargs:
            n_epochs: {{ n_epochs }}
            lr: {{ lr }}
            early_stop: {{ early_stop }}
            batch_size: {{ batch_size }}
            weight_decay: {{ weight_decay }}
            metric: loss
            loss: mse
            n_jobs: 20
            GPU: 0
            pt_model_uri: "model.model_cls"
            pt_model_kwargs: {
                "num_features": 20,
                {% if num_timesteps %}num_timesteps: {{ num_timesteps }}{% endif %}
            }
    dataset:
        class: {{ dataset_cls | default("DatasetH") }}
        module_path: quant_master.data.dataset
        kwargs:
            handler:
                class: DataHandlerLP
                module_path: quant_master.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [{{ train_start | default("2008-01-01", true) }}, {{ train_end | default("2014-12-31", true) }}]
                valid: [{{ valid_start | default("2015-01-01", true) }}, {{ valid_end | default("2016-12-31", true) }}]
                test: [{{ test_start | default("2017-01-01", true) }}, {{ test_end | default("null", true) }}]
            {% if step_len %}step_len: {{ step_len }}{% endif %}
    record:
        - class: SignalRecord
          module_path: quant_master.workflow.record_temp
          kwargs:
            model: <MODEL>
            dataset: <DATASET>
        - class: SigAnaRecord
          module_path: quant_master.workflow.record_temp
          kwargs:
            ana_long_short: False
            ann_scaler: 252
        - class: PortAnaRecord
          module_path: quant_master.workflow.record_temp
          kwargs:
            config: *port_analysis_config
"""
