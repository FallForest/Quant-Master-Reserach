#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import quant_master
from quant_master.constant import REG_CN

from quant_master.utils import init_instance_by_config
from quant_master.tests.data import GetData
from quant_master.tests.config import CSI300_GBDT_TASK

if __name__ == "__main__":
    # use default data
    provider_uri = "~/.quant_master/quant_master_data/cn_data"  # target_dir
    GetData().quant_master_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)

    quant_master.init(provider_uri=provider_uri, region=REG_CN)

    ###################################
    # train model
    ###################################
    # model initialization
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    model.fit(dataset)

    # get model feature importance
    feature_importance = model.get_feature_importance()
    print("feature importance:")
    print(feature_importance)
