import os
from pathlib import Path
from typing import Optional

import fire
import pandas as pd
from loguru import logger
from tqdm import tqdm

import quant_master
from quant_master.data import D


class DataHealthChecker:
    """Checks a dataset for data completeness and correctness. The data will be converted to a pd.DataFrame and checked for the following problems:
    - any of the columns ["open", "high", "low", "close", "volume"] are missing
    - any data is missing
    - any step change in the OHLCV columns is above a threshold (default: 0.5 for price, 3 for volume)
    - any factor is missing
    """

    def __init__(
        self,
        csv_path=None,
        quant_master_dir=None,
        freq="day",
        large_step_threshold_price=0.5,
        large_step_threshold_volume=3,
        missing_data_num=0,
        require_nonconstant_factor=False,
        constant_factor_ratio_threshold=0.95,
        universe=None,
        universe_check_date=None,
        max_active_universe_size=None,
        fail_fast=False,
    ):
        assert csv_path or quant_master_dir, "One of csv_path or quant_master_dir should be provided."
        assert not (csv_path and quant_master_dir), "Only one of csv_path or quant_master_dir should be provided."

        self.data = {}
        self.problems = {}
        self.freq = freq
        self.large_step_threshold_price = large_step_threshold_price
        self.large_step_threshold_volume = large_step_threshold_volume
        self.missing_data_num = missing_data_num
        self.require_nonconstant_factor = require_nonconstant_factor
        self.constant_factor_ratio_threshold = constant_factor_ratio_threshold
        self.universe = universe
        self.universe_check_date = universe_check_date
        self.max_active_universe_size = max_active_universe_size
        self.fail_fast = fail_fast
        self.quant_master_dir = os.path.abspath(os.path.expanduser(quant_master_dir)) if quant_master_dir else None

        if csv_path:
            assert os.path.isdir(csv_path), f"{csv_path} should be a directory."
            files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
            for filename in tqdm(files, desc="Loading data"):
                df = pd.read_csv(os.path.join(csv_path, filename))
                self.data[filename] = df

        elif quant_master_dir:
            quant_master.init(provider_uri=quant_master_dir)
            self.load_quant_master_data()

    def load_quant_master_data(self):
        instruments = D.instruments(market="all")
        instrument_list = D.list_instruments(instruments=instruments, as_list=True, freq=self.freq)
        required_fields = ["$open", "$close", "$low", "$high", "$volume", "$factor"]
        for instrument in instrument_list:
            df = D.features([instrument], required_fields, freq=self.freq)
            df.rename(
                columns={
                    "$open": "open",
                    "$close": "close",
                    "$low": "low",
                    "$high": "high",
                    "$volume": "volume",
                    "$factor": "factor",
                },
                inplace=True,
            )
            self.data[instrument] = df

    # NOTE:
    # This check is added due to a known issue in QuantMaster where feature paths
    # are constructed using lowercased instrument names. On case-sensitive
    # file systems (e.g. Linux), uppercase directory names under `features/`
    # will cause data loading failures.
    #
    # See: https://github.com/microsoft/quant_master/issues/2053
    def check_features_dir_lowercase(self) -> Optional[pd.DataFrame]:
        """
        Check whether all subdirectories under `<quant_master_dir>/features` are named in lowercase.

        This validation helps prevent data loading issues on case-sensitive
        file systems caused by uppercase instrument directory names.
        """
        if not self.quant_master_dir:
            return None

        features_dir = os.path.join(self.quant_master_dir, "features")
        if not os.path.isdir(features_dir):
            logger.warning(f"`features` directory not found under {self.quant_master_dir}")
            return None

        bad_dirs = []
        for name in os.listdir(features_dir):
            full_path = os.path.join(features_dir, name)
            if os.path.isdir(full_path) and name != name.lower():
                bad_dirs.append(name)

        if bad_dirs:
            result_df = pd.DataFrame({"non_lowercase_dir": bad_dirs})
            return result_df
        else:
            logger.info(
                f"✅ All subdirectories under `{os.path.join(self.quant_master_dir, 'features')}` are named in lowercase."
            )
            return None

    def check_missing_data(self) -> Optional[pd.DataFrame]:
        """Check if any data is missing in the DataFrame."""
        result_dict = {
            "instruments": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        for filename, df in self.data.items():
            missing_data_columns = df.isnull().sum()[df.isnull().sum() > self.missing_data_num].index.tolist()
            if len(missing_data_columns) > 0:
                result_dict["instruments"].append(filename)
                result_dict["open"].append(df.isnull().sum()["open"])
                result_dict["high"].append(df.isnull().sum()["high"])
                result_dict["low"].append(df.isnull().sum()["low"])
                result_dict["close"].append(df.isnull().sum()["close"])
                result_dict["volume"].append(df.isnull().sum()["volume"])

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ There are no missing data.")
            return None

    def check_large_step_changes(self) -> Optional[pd.DataFrame]:
        """Check if there are any large step changes above the threshold in the OHLCV columns."""
        result_dict = {
            "instruments": [],
            "col_name": [],
            "date": [],
            "pct_change": [],
        }
        for filename, df in self.data.items():
            affected_columns = []
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    pct_change = df[col].pct_change(fill_method=None).abs()
                    threshold = self.large_step_threshold_volume if col == "volume" else self.large_step_threshold_price
                    if pct_change.max() > threshold:
                        large_steps = pct_change[pct_change > threshold]
                        result_dict["instruments"].append(filename)
                        result_dict["col_name"].append(col)
                        result_dict["date"].append(large_steps.index.to_list()[0][1].strftime("%Y-%m-%d"))
                        result_dict["pct_change"].append(pct_change.max())
                        affected_columns.append(col)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ There are no large step changes in the OHLCV column above the threshold.")
            return None

    def check_required_columns(self) -> Optional[pd.DataFrame]:
        """Check if any of the required columns (OLHCV) are missing in the DataFrame."""
        required_columns = ["open", "high", "low", "close", "volume"]
        result_dict = {
            "instruments": [],
            "missing_col": [],
        }
        for filename, df in self.data.items():
            if not all(column in df.columns for column in required_columns):
                missing_required_columns = [column for column in required_columns if column not in df.columns]
                result_dict["instruments"].append(filename)
                result_dict["missing_col"] += missing_required_columns

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ The columns (OLHCV) are complete and not missing.")
            return None

    def check_missing_factor(self) -> Optional[pd.DataFrame]:
        """Check if the 'factor' column is missing in the DataFrame."""
        result_dict = {
            "instruments": [],
            "missing_factor_col": [],
            "missing_factor_data": [],
        }
        for filename, df in self.data.items():
            if "000300" in filename or "000903" in filename or "000905" in filename:
                continue
            if "factor" not in df.columns:
                result_dict["instruments"].append(filename)
                result_dict["missing_factor_col"].append(True)
                result_dict["missing_factor_data"].append(False)
                continue
            if df["factor"].isnull().all():
                if filename in result_dict["instruments"]:
                    result_dict["missing_factor_data"].append(True)
                else:
                    result_dict["instruments"].append(filename)
                    result_dict["missing_factor_col"].append(False)
                    result_dict["missing_factor_data"].append(True)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ The `factor` column already exists and is not empty.")
            return None

    def check_constant_factor(self) -> Optional[pd.DataFrame]:
        """Report instruments whose available adjustment factor is always one."""
        if not self.require_nonconstant_factor:
            return None

        eligible = 0
        constant_one = 0
        for instrument, df in self.data.items():
            if "factor" not in df.columns:
                continue
            values = pd.to_numeric(df["factor"], errors="coerce").dropna()
            if values.empty:
                continue
            eligible += 1
            if not values.empty and values.eq(1.0).all():
                constant_one += 1
        ratio = constant_one / eligible if eligible else 0.0
        if eligible and ratio >= float(self.constant_factor_ratio_threshold):
            return pd.DataFrame(
                [
                    {
                        "checked_instruments": eligible,
                        "constant_one_instruments": constant_one,
                        "constant_one_ratio": ratio,
                    }
                ],
                index=["factor"],
            )
        logger.info(f"Always-one adjustment factor ratio is {ratio:.2%} ({constant_one}/{eligible}).")
        return None

    def check_active_universe_size(self) -> Optional[pd.DataFrame]:
        """Check one universe snapshot from the read-only instruments file."""
        if self.max_active_universe_size is None:
            return None
        if not self.quant_master_dir or not self.universe:
            raise ValueError("`quant_master_dir` and `universe` are required for the universe-size check.")

        instrument_path = Path(self.quant_master_dir) / "instruments" / f"{self.universe.lower()}.txt"
        if not instrument_path.is_file():
            raise FileNotFoundError(f"Universe file does not exist: {instrument_path}")
        frame = pd.read_csv(instrument_path, sep="\t", names=["instrument", "start_time", "end_time"])
        frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
        frame["end_time"] = pd.to_datetime(frame["end_time"], errors="coerce")
        frame = frame.dropna(subset=["instrument", "start_time", "end_time"])
        if frame.empty:
            raise ValueError(f"Universe file has no valid intervals: {instrument_path}")

        check_date = pd.Timestamp(self.universe_check_date) if self.universe_check_date else frame["end_time"].max()
        active = frame.loc[
            (frame["start_time"] <= check_date) & (frame["end_time"] >= check_date), "instrument"
        ].nunique()
        if active > int(self.max_active_universe_size):
            return pd.DataFrame(
                [{"universe": self.universe, "date": check_date.date(), "active_instruments": active}]
            ).set_index("universe")
        logger.info(f"{self.universe} has {active} active instruments on {check_date.date()}.")
        return None

    def check_data(self):
        check_missing_data_result = self.check_missing_data()
        check_large_step_changes_result = self.check_large_step_changes()
        check_required_columns_result = self.check_required_columns()
        check_missing_factor_result = self.check_missing_factor()
        check_features_dir_case_result = self.check_features_dir_lowercase()
        check_constant_factor_result = self.check_constant_factor()
        check_universe_size_result = self.check_active_universe_size()
        results = {
            "missing_data": check_missing_data_result,
            "large_step_changes": check_large_step_changes_result,
            "required_columns": check_required_columns_result,
            "missing_factor": check_missing_factor_result,
            "features_dir_case": check_features_dir_case_result,
            "constant_factor": check_constant_factor_result,
            "active_universe_size": check_universe_size_result,
        }
        if (
            check_missing_data_result is not None
            or check_large_step_changes_result is not None
            or check_required_columns_result is not None
            or check_missing_factor_result is not None
            or check_features_dir_case_result is not None
            or check_constant_factor_result is not None
            or check_universe_size_result is not None
        ):
            print(f"\nSummary of data health check ({len(self.data)} files checked):")
            print("-------------------------------------------------")
            if isinstance(check_missing_data_result, pd.DataFrame):
                logger.warning(f"There is missing data.")
                print(check_missing_data_result)
            if isinstance(check_large_step_changes_result, pd.DataFrame):
                logger.warning(f"The OHLCV column has large step changes.")
                print(check_large_step_changes_result)
            if isinstance(check_required_columns_result, pd.DataFrame):
                logger.warning(f"Columns (OLHCV) are missing.")
                print(check_required_columns_result)
            if isinstance(check_missing_factor_result, pd.DataFrame):
                logger.warning(f"The factor column does not exist or is empty")
                print(check_missing_factor_result)
            if isinstance(check_features_dir_case_result, pd.DataFrame):
                logger.warning(
                    f"Some subdirectories under `{os.path.join(self.quant_master_dir, 'features')}` contain uppercase letters, please rename them to lowercase manually."
                )
                print(check_features_dir_case_result)
            if isinstance(check_constant_factor_result, pd.DataFrame):
                logger.warning("Adjustment factor is always one for some instruments; prices may be unadjusted.")
                print(check_constant_factor_result)
            if isinstance(check_universe_size_result, pd.DataFrame):
                logger.warning("The active universe exceeds the configured maximum size.")
                print(check_universe_size_result)
            if self.fail_fast:
                failed = [name for name, result in results.items() if result is not None]
                raise RuntimeError(f"Data health check failed: {', '.join(failed)}")
        return results


if __name__ == "__main__":
    fire.Fire(DataHealthChecker)
