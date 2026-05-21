.. _installation:

============
Installation
============

.. currentmodule:: quant_master


``QuantMaster`` Installation
=====================
.. note::

   `QuantMaster` supports both `Windows` and `Linux`. It's recommended to use `QuantMaster` in `Linux`. ``QuantMaster`` supports Python3, which is up to Python3.8.

Users can easily install ``QuantMaster`` by pip according to the following command:

.. code-block:: bash

   pip install pyquant_master


Also, Users can install ``QuantMaster`` by the source code according to the following steps:

- Enter the root directory of ``QuantMaster``, in which the file ``setup.py`` exists.
- Then, please execute the following command to install the environment dependencies and install ``QuantMaster``:

   .. code-block:: bash

      $ pip install numpy
      $ pip install --upgrade cython
      $ git clone https://github.com/microsoft/quant_master.git && cd quant_master
      $ python setup.py install

.. note::
   It's recommended to use anaconda/miniconda to setup the environment. ``QuantMaster`` needs lightgbm and pytorch packages, use pip to install them.



Use the following code to make sure the installation successful:

.. code-block:: python

   >>> import quant_master
   >>> quant_master.__version__
   <LATEST VERSION>
