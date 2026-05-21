.. _server:

=============================
``Online`` & ``Offline`` mode
=============================
.. currentmodule:: quant_master


Introduction
============

``QuantMaster`` supports ``Online`` mode and ``Offline`` mode. Only the ``Offline`` mode is introduced in this document.

The ``Online`` mode is designed to solve the following problems:

- Manage the data in a centralized way. Users don't have to manage data of different versions.
- Reduce the amount of cache to be generated.
- Make the data can be accessed in a remote way.

QuantMaster-Server
===========

``QuantMaster-Server`` is the assorted server system for ``QuantMaster``, which utilizes ``QuantMaster`` for basic calculations and provides extensive server system and cache mechanism. With QLibServer, the data provided for ``QuantMaster`` can be managed in a centralized manner. With ``QuantMaster-Server``, users can use ``QuantMaster`` in ``Online`` mode.



Reference
=========
If users are interested in ``QuantMaster-Server`` and ``Online`` mode, please refer to `QuantMaster-Server Project <https://github.com/microsoft/quant_master-server>`_ and `QuantMaster-Server Document <https://quant_master-server.readthedocs.io/en/latest/>`_.
