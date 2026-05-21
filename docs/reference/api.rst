.. _api:

=============
API Reference
=============



Here you can find all ``QuantMaster`` interfaces.


Data
====

Provider
--------

.. automodule:: quant_master.data.data
    :members:

Filter
------

.. automodule:: quant_master.data.filter
    :members:

Class
-----
.. automodule:: quant_master.data.base
    :members:

Operator
--------
.. automodule:: quant_master.data.ops
    :members:

Cache
-----
.. autoclass:: quant_master.data.cache.MemCacheUnit
    :members:

.. autoclass:: quant_master.data.cache.MemCache
    :members:

.. autoclass:: quant_master.data.cache.ExpressionCache
    :members:

.. autoclass:: quant_master.data.cache.DatasetCache
    :members:

.. autoclass:: quant_master.data.cache.DiskExpressionCache
    :members:

.. autoclass:: quant_master.data.cache.DiskDatasetCache
    :members:


Storage
-------
.. autoclass:: quant_master.data.storage.storage.BaseStorage
    :members:

.. autoclass:: quant_master.data.storage.storage.CalendarStorage
    :members:

.. autoclass:: quant_master.data.storage.storage.InstrumentStorage
    :members:

.. autoclass:: quant_master.data.storage.storage.FeatureStorage
    :members:

.. autoclass:: quant_master.data.storage.file_storage.FileStorageMixin
    :members:

.. autoclass:: quant_master.data.storage.file_storage.FileCalendarStorage
    :members:

.. autoclass:: quant_master.data.storage.file_storage.FileInstrumentStorage
    :members:

.. autoclass:: quant_master.data.storage.file_storage.FileFeatureStorage
    :members:


Dataset
-------

Dataset Class
~~~~~~~~~~~~~
.. automodule:: quant_master.data.dataset.__init__
    :members:

Data Loader
~~~~~~~~~~~
.. automodule:: quant_master.data.dataset.loader
    :members:

Data Handler
~~~~~~~~~~~~
.. automodule:: quant_master.data.dataset.handler
    :members:

Processor
~~~~~~~~~
.. automodule:: quant_master.data.dataset.processor
    :members:


Contrib
=======

Model
-----
.. automodule:: quant_master.model.base
    :members:

Strategy
--------

.. automodule:: quant_master.contrib.strategy
    :members:

Evaluate
--------

.. automodule:: quant_master.contrib.evaluate
    :members:


Report
------

.. automodule:: quant_master.contrib.report.analysis_position.report
    :members:



.. automodule:: quant_master.contrib.report.analysis_position.score_ic
    :members:



.. automodule:: quant_master.contrib.report.analysis_position.cumulative_return
    :members:



.. automodule:: quant_master.contrib.report.analysis_position.risk_analysis
    :members:



.. automodule:: quant_master.contrib.report.analysis_position.rank_label
    :members:



.. automodule:: quant_master.contrib.report.analysis_model.analysis_model_performance
    :members:


Workflow
========


Experiment Manager
------------------
.. autoclass:: quant_master.workflow.expm.ExpManager
    :members:

Experiment
----------
.. autoclass:: quant_master.workflow.exp.Experiment
    :members:

Recorder
--------
.. autoclass:: quant_master.workflow.recorder.Recorder
    :members:

Record Template
---------------
.. automodule:: quant_master.workflow.record_temp
    :members:

Task Management
===============


TaskGen
-------
.. automodule:: quant_master.workflow.task.gen
    :members:

TaskManager
-----------
.. automodule:: quant_master.workflow.task.manage
    :members:

Trainer
-------
.. automodule:: quant_master.model.trainer
    :members:

Collector
---------
.. automodule:: quant_master.workflow.task.collect
    :members:

Group
-----
.. automodule:: quant_master.model.ens.group
    :members:

Ensemble
--------
.. automodule:: quant_master.model.ens.ensemble
    :members:

Utils
-----
.. automodule:: quant_master.workflow.task.utils
    :members:


Online Serving
==============


Online Manager
--------------
.. automodule:: quant_master.workflow.online.manager
    :members:

Online Strategy
---------------
.. automodule:: quant_master.workflow.online.strategy
    :members:

Online Tool
-----------
.. automodule:: quant_master.workflow.online.utils
    :members:


RecordUpdater
-------------
.. automodule:: quant_master.workflow.online.update
    :members:


Utils
=====

Serializable
------------

.. automodule:: quant_master.utils.serial
    :members:

RL
==============

Base Component
--------------
.. automodule:: quant_master.rl
    :members:
    :imported-members:

Strategy
--------
.. automodule:: quant_master.rl.strategy
    :members:
    :imported-members:

Trainer
-------
.. automodule:: quant_master.rl.trainer
    :members:
    :imported-members:

Order Execution
---------------
.. automodule:: quant_master.rl.order_execution
    :members:
    :imported-members:

Utils
---------------
.. automodule:: quant_master.rl.utils
    :members:
    :imported-members: