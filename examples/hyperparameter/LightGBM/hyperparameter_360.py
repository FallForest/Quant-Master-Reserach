import quant_master
import optuna
from quant_master.constant import REG_CN
from quant_master.utils import init_instance_by_config
from quant_master.tests.data import GetData
from quant_master.tests.config import get_dataset_config, CSI300_MARKET, DATASET_ALPHA360_CLASS

DATASET_CONFIG = get_dataset_config(market=CSI300_MARKET, dataset_class=DATASET_ALPHA360_CLASS)


def objective(trial):
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "quant_master.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
                "learning_rate": trial.suggest_uniform("learning_rate", 0, 1),
                "subsample": trial.suggest_uniform("subsample", 0, 1),
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e4),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e4),
                "max_depth": 10,
                "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            },
        },
    }

    evals_result = dict()
    model = init_instance_by_config(task["model"])
    model.fit(dataset, evals_result=evals_result)
    return min(evals_result["valid"])


if __name__ == "__main__":
    provider_uri = "~/.quant_master/quant_master_data/cn_data"
    GetData().quant_master_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    quant_master.init(provider_uri=provider_uri, region=REG_CN)

    dataset = init_instance_by_config(DATASET_CONFIG)

    study = optuna.Study(study_name="LGBM_360", storage="sqlite:///db.sqlite3")
    study.optimize(objective, n_jobs=6)
