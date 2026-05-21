#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import unittest

import quant_master
from quant_master.data import D
from quant_master.tests import TestAutoData
from multiprocessing import Pool


def get_features(fields):
    quant_master.init(provider_uri=TestAutoData.provider_uri, expression_cache=None, dataset_cache=None, joblib_backend="loky")
    return D.features(D.instruments("csi300"), fields)


class TestGetData(TestAutoData):
    FIELDS = "$open,$close,$high,$low,$volume,$factor,$change".split(",")

    def test_multi_proc(self):
        """
        For testing if it will raise error
        """
        iter_n = 2
        pool = Pool(iter_n)

        res = []
        for _ in range(iter_n):
            res.append(pool.apply_async(get_features, (self.FIELDS,), {}))

        for r in res:
            print(r.get())

        pool.close()
        pool.join()


if __name__ == "__main__":
    unittest.main()
