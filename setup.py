import os

import numpy
from setuptools import Extension, setup

NUMPY_INCLUDE = numpy.get_include()


setup(
    ext_modules=[
        Extension(
            "quant_master.data._libs.rolling",
            ["quant_master/data/_libs/rolling.pyx"],
            language="c++",
            include_dirs=[NUMPY_INCLUDE],
        ),
        Extension(
            "quant_master.data._libs.expanding",
            ["quant_master/data/_libs/expanding.pyx"],
            language="c++",
            include_dirs=[NUMPY_INCLUDE],
        ),
    ],
)
