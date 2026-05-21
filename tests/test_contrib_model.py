# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import inspect

from quant_master.contrib.model import all_model_classes


class TestAllFlow(unittest.TestCase):
    def test_0_initialize(self):
        num = 0
        for model_class in all_model_classes:
            if model_class is not None:
                params = list(inspect.signature(model_class).parameters.values())
                required = [
                    p
                    for p in params
                    if p.default is inspect._empty
                    and p.kind
                    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                if required:
                    continue
                model = model_class()
                num += 1
        print("There are {:}/{:} valid models in total.".format(num, len(all_model_classes)))


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow("test_0_initialize"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
