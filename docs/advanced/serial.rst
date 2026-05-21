.. _serial:

=============
Serialization
=============
.. currentmodule:: quant_master

Introduction
============
``QuantMaster`` supports dumping the state of ``DataHandler``, ``DataSet``, ``Processor`` and ``Model``, etc. into a disk and reloading them.

Serializable Class
==================

``QuantMaster`` provides a base class ``quant_master.utils.serial.Serializable``, whose state can be dumped into or loaded from disk in `pickle` format.
When users dump the state of a ``Serializable`` instance, the attributes of the instance whose name **does not** start with `_` will be saved on the disk.
However, users can use ``config`` method or override ``default_dump_all`` attribute to prevent this feature.

Users can also override ``pickle_backend`` attribute to choose a pickle backend. The supported value is "pickle" (default and common) and "dill" (dump more things such as function, more information in `here <https://pypi.org/project/dill/>`_).

Example
=======
``QuantMaster``'s serializable class includes  ``DataHandler``, ``DataSet``, ``Processor`` and ``Model``, etc., which are subclass of  ``quant_master.utils.serial.Serializable``.
Specifically, ``quant_master.data.dataset.DatasetH`` is one of them. Users can serialize ``DatasetH`` as follows.

.. code-block:: Python

    ##=============dump dataset=============
    dataset.to_pickle(path="dataset.pkl") # dataset is an instance of quant_master.data.dataset.DatasetH

    ##=============reload dataset=============
    with open("dataset.pkl", "rb") as file_dataset:
        dataset = pickle.load(file_dataset)

.. note::
    Only state of ``DatasetH`` should be saved on the disk, such as some `mean` and `variance` used for data normalization, etc.

    After reloading the ``DatasetH``, users need to reinitialize it. It means that users can reset some states of ``DatasetH`` or ``QuantMasterDataHandler`` such as `instruments`, `start_time`, `end_time` and `segments`, etc.,  and generate new data according to the states (data is not state and should not be saved on the disk).

A more detailed example is in this `link <https://github.com/microsoft/quant_master/tree/main/examples/highfreq>`_.


API
===
Please refer to `Serializable API <../reference/api.html#module-quant_master.utils.serial.Serializable>`_.
