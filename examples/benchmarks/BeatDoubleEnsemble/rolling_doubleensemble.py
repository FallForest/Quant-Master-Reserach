# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Union

import fire

from qlib import auto_init
from qlib.contrib.rolling.base import Rolling

DIRNAME = Path(__file__).absolute().resolve().parent


class RollingDoubleEnsemble(Rolling):
    DEFAULT_CONF = DIRNAME / "workflow_config_doubleensemble_rolling_Alpha158_2026_local.yaml"

    def __init__(
        self,
        conf_path: Union[str, Path] = DEFAULT_CONF,
        horizon: int = 1,
        step: int = 252,
        rolling_exp: str = None,
        **kwargs,
    ) -> None:
        conf_path = Path(conf_path)
        super().__init__(
            conf_path=conf_path,
            horizon=horizon,
            step=step,
            rolling_exp=rolling_exp,
            **kwargs,
        )

    def basic_task(self, enable_handler_cache: bool = False):
        # The local security policy blocks unpickling cached handlers in rolling mode.
        # Use the raw handler config directly for this benchmark.
        return super().basic_task(enable_handler_cache=enable_handler_cache)


if __name__ == "__main__":
    auto_init()
    fire.Fire(RollingDoubleEnsemble)
