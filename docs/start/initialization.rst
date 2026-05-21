.. _initialization:

===================
QuantMaster Initialization
===================

.. currentmodule:: quant_master


Initialization
==============

Please follow the steps below to initialize ``QuantMaster``.

Download and prepare the Data: execute the following command to download stock data. Please pay `attention` that the data is collected from `Yahoo Finance <https://finance.yahoo.com/lookup>`_ and the data might not be perfect. We recommend users to prepare their own data if they have high-quality datasets. Please refer to `Data <../component/data.html#converting-csv-format-into-quant_master-format>`_ for more information about customized dataset.

    .. code-block:: bash

        python scripts/get_data.py quant_master_data --target_dir ~/.quant_master/quant_master_data/cn_data --region cn

Please refer to `Data Preparation <../component/data.html#data-preparation>`_ for more information about `get_data.py`,


Initialize QuantMaster before calling other APIs: run following code in python.

    .. code-block:: Python

        import quant_master
        # region in [REG_CN, REG_US]
        from quant_master.constant import REG_CN
        provider_uri = "~/.quant_master/quant_master_data/cn_data"  # target_dir
        quant_master.init(provider_uri=provider_uri, region=REG_CN)

.. note::
   Do not import quant_master package in the repository directory  of ``QuantMaster``, otherwise, errors may occur.

Parameters
-------------------

Besides `provider_uri` and `region`, `quant_master.init` has other parameters.
The following are several important parameters of `quant_master.init` (`QuantMaster` has a lot of config. Only part of parameters are limited here. More detailed setting can be found `here <https://github.com/microsoft/quant_master/blob/main/quant_master/config.py>`_):

- `provider_uri`
    Type: str. The URI of the QuantMaster data. For example, it could be the location where the data loaded by ``get_data.py`` are stored.
- `region`
    Type: str, optional parameter(default: `quant_master.constant.REG_CN`).
        Currently: ``quant_master.constant.REG_US`` ('us') and ``quant_master.constant.REG_CN`` ('cn') is supported. Different value of  `region` will result in different stock market mode.
        - ``quant_master.constant.REG_US``: US stock market.
        - ``quant_master.constant.REG_CN``: China stock market.

        Different modes will result in different trading limitations and costs.
        The region is just `shortcuts for defining a batch of configurations <https://github.com/microsoft/quant_master/blob/528f74af099bf6156e9480bcd2bb28e453231212/quant_master/config.py#L249>`_, which include minimal trading order unit (``trade_unit``),  trading limitation (``limit_threshold``) , etc.  It is not a necessary part and users can set the key configurations manually if the existing region setting can't meet their requirements.
- `redis_host`
    Type: str, optional parameter(default: "127.0.0.1"), host of `redis`
        The lock and cache mechanism relies on redis.
- `redis_port`
    Type: int, optional parameter(default: 6379), port of `redis`

    .. note::

        The value of `region` should be aligned with the data stored in `provider_uri`. Currently, ``scripts/get_data.py`` only provides China stock market data. If users want to use the US stock market data, they should prepare their own US-stock data in `provider_uri` and switch to US-stock mode.

    .. note::

        If QuantMaster fails to connect redis via `redis_host` and `redis_port`, cache mechanism will not be used! Please refer to `Cache <../component/data.html#cache>`_ for details.
- `exp_manager`
    Type: dict, optional parameter, the setting of `experiment manager` to be used in quant_master. Users can specify an experiment manager class, as well as the tracking URI for all the experiments. However, please be aware that we only support input of a dictionary in the following style for `exp_manager`. For more information about `exp_manager`, users can refer to `Recorder: Experiment Management <../component/recorder.html>`_.

    .. code-block:: Python

        # For example, if you want to set your tracking_uri to a <specific folder>, you can initialize quant_master below
        quant_master.init(provider_uri=provider_uri, region=REG_CN, exp_manager= {
            "class": "MLflowExpManager",
            "module_path": "quant_master.workflow.expm",
            "kwargs": {
                "uri": "python_execution_path/mlruns",
                "default_exp_name": "Experiment",
            }
        })
- `mongo`
    Type: dict, optional parameter, the setting of `MongoDB <https://www.mongodb.com/>`_ which will be used in some features such as `Task Management <../advanced/task_management.html>`_, with high performance and clustered processing.
    Users need to follow the steps in  `installation <https://www.mongodb.com/try/download/community>`_  to install MongoDB firstly and then access it via a URI.
    Users can access mongodb with credential by setting "task_url"  to a string like `"mongodb://%s:%s@%s" % (user, pwd, host + ":" + port)`.

    .. code-block:: Python

        # For example, you can initialize quant_master below
        quant_master.init(provider_uri=provider_uri, region=REG_CN, mongo={
            "task_url": "mongodb://localhost:27017/",  # your mongo url
            "task_db_name": "rolling_db", # the database name of Task Management
        })

- `logging_level`
    The logging level for the system.

- `kernels`
    The number of processes used when calculating features in QuantMaster's expression engine. It is very helpful to set it to 1 when you are debuggin an expression calculating exception
