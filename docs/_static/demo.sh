#!/bin/sh
git clone https://github.com/microsoft/quant_master.git
cd quant_master
ls
pip install pyquant_master
# or
# pip install numpy
# pip install --upgrade cython
# python setup.py install
cd examples
ls
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml