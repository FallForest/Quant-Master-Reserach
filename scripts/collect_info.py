import sys
import platform
import quant_master
import fire
import pkg_resources
from pathlib import Path

QUANT_MASTER_PATH = Path(__file__).absolute().resolve().parent.parent


class InfoCollector:
    """
    User could collect system info by following commands
    `cd scripts && python collect_info.py all`
    - NOTE: please avoid running this script in the project folder which contains `quant_master`
    """

    def sys(self):
        """collect system related info"""
        for method in ["system", "machine", "platform", "version"]:
            print(getattr(platform, method)())

    def py(self):
        """collect Python related info"""
        print("Python version: {}".format(sys.version.replace("\n", " ")))

    def quant_master(self):
        """collect quant_master related info"""
        print("QuantMaster version: {}".format(quant_master.__version__))
        REQUIRED = [
            "setuptools",
            "wheel",
            "cython",
            "pyyaml",
            "numpy",
            "pandas",
            "mlflow",
            "filelock",
            "redis",
            "dill",
            "fire",
            "ruamel.yaml",
            "python-redis-lock",
            "tqdm",
            "pymongo",
            "loguru",
            "lightgbm",
            "gym",
            "cvxpy",
            "joblib",
            "matplotlib",
            "jupyter",
            "nbconvert",
            "pyarrow",
            "pydantic-settings",
            "setuptools-scm",
        ]

        for package in REQUIRED:
            version = pkg_resources.get_distribution(package).version
            print(f"{package}=={version}")

    def all(self):
        """collect all info"""
        for method in ["sys", "py", "quant_master"]:
            getattr(self, method)()
            print()


if __name__ == "__main__":
    fire.Fire(InfoCollector)
