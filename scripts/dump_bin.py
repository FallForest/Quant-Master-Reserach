# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import shutil
import traceback
from pathlib import Path
from typing import Iterable, List, Union
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from quant_master.utils import fname_to_code, code_to_fname


def read_as_df(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read a csv or parquet file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the data file.
    **kwargs :
        Additional keyword arguments passed to the underlying pandas
        reader.

    Returns
    -------
    pd.DataFrame
    """
    file_path = Path(file_path).expanduser()
    suffix = file_path.suffix.lower()

    keep_keys = {".csv": ("low_memory",)}
    kept_kwargs = {}
    for k in keep_keys.get(suffix, []):
        if k in kwargs:
            kept_kwargs[k] = kwargs[k]

    if suffix == ".csv":
        return pd.read_csv(file_path, **kept_kwargs)
    elif suffix == ".parquet":
        return pd.read_parquet(file_path, **kept_kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


class DumpDataBase:
    INSTRUMENTS_START_FIELD = "start_datetime"
    INSTRUMENTS_END_FIELD = "end_datetime"
    CALENDARS_DIR_NAME = "calendars"
    FEATURES_DIR_NAME = "features"
    INSTRUMENTS_DIR_NAME = "instruments"
    DUMP_FILE_SUFFIX = ".bin"
    DAILY_FORMAT = "%Y-%m-%d"
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"
    INSTRUMENTS_SEP = "\t"
    INSTRUMENTS_FILE_NAME = "all.txt"

    UPDATE_MODE = "update"
    ALL_MODE = "all"

    def __init__(
        self,
        data_path: str,
        quant_master_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        data_path: str
            stock data path or directory
        quant_master_dir: str
            quant_master(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup quant_master_dir to backup_dir
        freq: str, default "day"
            transaction frequency
        max_workers: int, default None
            number of threads
        date_field_name: str, default "date"
            the name of the date field in the csv
        file_suffix: str, default ".csv"
            file suffix
        symbol_field_name: str, default "symbol"
            symbol field name
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        """
        data_path = Path(data_path).expanduser()
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        self.file_suffix = file_suffix
        self.symbol_field_name = symbol_field_name
        self.df_files = sorted(data_path.glob(f"*{self.file_suffix}") if data_path.is_dir() else [data_path])
        if limit_nums is not None:
            self.df_files = self.df_files[: int(limit_nums)]
        self.quant_master_dir = Path(quant_master_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        if backup_dir is not None:
            self._backup_quant_master_dir(Path(backup_dir).expanduser())

        self.freq = freq
        self.calendar_format = self.DAILY_FORMAT if self.freq == "day" else self.HIGH_FREQ_FORMAT

        resolved_workers = 8 if max_workers is None else max(int(max_workers), 8)
        self.works = resolved_workers
        self.date_field_name = date_field_name

        self._calendars_dir = self.quant_master_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.quant_master_dir.joinpath(self.FEATURES_DIR_NAME)
        self._instruments_dir = self.quant_master_dir.joinpath(self.INSTRUMENTS_DIR_NAME)

        self._calendars_list = []

        self._mode = self.ALL_MODE
        self._kwargs = {}

    def _backup_quant_master_dir(self, target_dir: Path):
        shutil.copytree(str(self.quant_master_dir.resolve()), str(target_dir.resolve()))

    def _format_datetime(self, datetime_d: [str, pd.Timestamp]):
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.calendar_format)

    def _get_date(
        self, file_or_df: [Path, pd.DataFrame], *, is_begin_end: bool = False, as_set: bool = False
    ) -> Iterable[pd.Timestamp]:
        if not isinstance(file_or_df, pd.DataFrame):
            df = self._get_source_data(file_or_df)
        else:
            df = file_or_df
        if df.empty or self.date_field_name not in df.columns.tolist():
            _calendars = pd.Series(dtype=np.float32)
        else:
            _calendars = df[self.date_field_name]

        if is_begin_end and as_set:
            return (_calendars.min(), _calendars.max()), set(_calendars)
        elif is_begin_end:
            return _calendars.min(), _calendars.max()
        elif as_set:
            return set(_calendars)
        else:
            return _calendars.tolist()

    def _get_source_data(self, file_path: Path) -> pd.DataFrame:
        df = read_as_df(file_path, low_memory=False)
        if self.date_field_name in df.columns:
            df[self.date_field_name] = pd.to_datetime(df[self.date_field_name])
        # df.drop_duplicates([self.date_field_name], inplace=True)
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        return fname_to_code(file_path.stem.strip().lower())

    def get_dump_fields(self, df_columns: Iterable[str]) -> Iterable[str]:
        return (
            self._include_fields
            if self._include_fields
            else set(df_columns) - set(self._exclude_fields) if self._exclude_fields else df_columns
        )

    @staticmethod
    def _read_calendars(calendar_path: Path) -> List[pd.Timestamp]:
        return sorted(
            map(
                pd.Timestamp,
                pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
            )
        )

    def _read_instruments(self, instrument_path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            instrument_path,
            sep=self.INSTRUMENTS_SEP,
            names=[
                self.symbol_field_name,
                self.INSTRUMENTS_START_FIELD,
                self.INSTRUMENTS_END_FIELD,
            ],
        )

        return df

    def save_calendars(self, calendars_data: list):
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        calendars_path = str(self._calendars_dir.joinpath(f"{self.freq}.txt").expanduser().resolve())
        result_calendars_list = [self._format_datetime(x) for x in calendars_data]
        np.savetxt(calendars_path, result_calendars_list, fmt="%s", encoding="utf-8")

    def save_instruments(self, instruments_data: Union[list, pd.DataFrame]):
        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        instruments_path = str(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME).resolve())
        if isinstance(instruments_data, pd.DataFrame):
            _df_fields = [self.symbol_field_name, self.INSTRUMENTS_START_FIELD, self.INSTRUMENTS_END_FIELD]
            instruments_data = instruments_data.loc[:, _df_fields]
            instruments_data[self.symbol_field_name] = instruments_data[self.symbol_field_name].apply(
                lambda x: fname_to_code(x.lower()).upper()
            )
            instruments_data.to_csv(instruments_path, header=False, sep=self.INSTRUMENTS_SEP, index=False)
        else:
            np.savetxt(instruments_path, instruments_data, fmt="%s", encoding="utf-8")

    def data_merge_calendar(self, df: pd.DataFrame, calendars_list: List[pd.Timestamp]) -> pd.DataFrame:
        # calendars
        calendars_df = pd.DataFrame(data=calendars_list, columns=[self.date_field_name])
        calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype("datetime64[ns]")
        cal_df = calendars_df[
            (calendars_df[self.date_field_name] >= df[self.date_field_name].min())
            & (calendars_df[self.date_field_name] <= df[self.date_field_name].max())
        ]
        # align index
        cal_df.set_index(self.date_field_name, inplace=True)
        df.set_index(self.date_field_name, inplace=True)
        r_df = df.reindex(cal_df.index)
        return r_df

    @staticmethod
    def get_datetime_index(df: pd.DataFrame, calendar_list: List[pd.Timestamp]) -> int:
        return calendar_list.index(df.index.min())

    def _data_to_bin(self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path):
        if df.empty:
            logger.warning(f"{features_dir.name} data is None or empty")
            return
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        # align index
        _df = self.data_merge_calendar(df, calendar_list)
        if _df.empty:
            logger.warning(f"{features_dir.name} data is not in calendars")
            return
        # used when creating a bin file
        date_index = self.get_datetime_index(_df, calendar_list)
        for field in self.get_dump_fields(_df.columns):
            bin_path = features_dir.joinpath(f"{field.lower()}.{self.freq}{self.DUMP_FILE_SUFFIX}")
            if field not in _df.columns:
                continue
            if bin_path.exists() and self._mode == self.UPDATE_MODE:
                # update
                with bin_path.open("ab") as fp:
                    np.array(_df[field]).astype("<f").tofile(fp)
            else:
                # append; self._mode == self.ALL_MODE or not bin_path.exists()
                np.hstack([date_index, _df[field]]).astype("<f").tofile(str(bin_path.resolve()))

    def _dump_bin(self, file_or_data: [Path, pd.DataFrame], calendar_list: List[pd.Timestamp]):
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        if isinstance(file_or_data, pd.DataFrame):
            if file_or_data.empty:
                return
            code = fname_to_code(str(file_or_data.iloc[0][self.symbol_field_name]).lower())
            df = file_or_data
        elif isinstance(file_or_data, Path):
            code = self.get_symbol_from_file(file_or_data)
            df = self._get_source_data(file_or_data)
        else:
            raise ValueError(f"not support {type(file_or_data)}")
        if df is None or df.empty:
            logger.warning(f"{code} data is None or empty")
            return

        # try to remove dup rows or it will cause exception when reindex.
        df = df.drop_duplicates(self.date_field_name)

        # features save dir
        features_dir = self._features_dir.joinpath(code_to_fname(code).lower())
        features_dir.mkdir(parents=True, exist_ok=True)
        self._data_to_bin(df, calendar_list, features_dir)

    @abc.abstractmethod
    def dump(self):
        raise NotImplementedError("dump not implemented!")

    def __call__(self, *args, **kwargs):
        self.dump()


class DumpDataAll(DumpDataBase):
    def _get_all_date(self):
        logger.info("start get all date......")
        all_datetime = set()
        date_range_list = []
        _fun = partial(self._get_date, as_set=True, is_begin_end=True)
        with tqdm(total=len(self.df_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for file_path, ((_begin_time, _end_time), _set_calendars) in zip(
                    self.df_files, executor.map(_fun, self.df_files)
                ):
                    all_datetime = all_datetime | _set_calendars
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(_end_time, pd.Timestamp):
                        _begin_time = self._format_datetime(_begin_time)
                        _end_time = self._format_datetime(_end_time)
                        symbol = self.get_symbol_from_file(file_path)
                        _inst_fields = [symbol.upper(), _begin_time, _end_time]
                        date_range_list.append(f"{self.INSTRUMENTS_SEP.join(_inst_fields)}")
                    p_bar.update()
        self._kwargs["all_datetime_set"] = all_datetime
        self._kwargs["date_range_list"] = date_range_list
        logger.info("end of get all date.\n")

    def _dump_calendars(self):
        logger.info("start dump calendars......")
        self._calendars_list = sorted(map(pd.Timestamp, self._kwargs["all_datetime_set"]))
        self.save_calendars(self._calendars_list)
        logger.info("end of calendars dump.\n")

    def _dump_instruments(self):
        logger.info("start dump instruments......")
        self.save_instruments(self._kwargs["date_range_list"])
        logger.info("end of instruments dump.\n")

    def _dump_features(self):
        logger.info("start dump features......")
        _dump_func = partial(self._dump_bin, calendar_list=self._calendars_list)
        with tqdm(total=len(self.df_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, self.df_files):
                    p_bar.update()

        logger.info("end of features dump.\n")

    def dump(self):
        self._get_all_date()
        self._dump_calendars()
        self._dump_instruments()
        self._dump_features()


class DumpDataFix(DumpDataAll):
    def _dump_instruments(self):
        logger.info("start dump instruments......")
        _fun = partial(self._get_date, is_begin_end=True)
        new_stock_files = sorted(
            filter(
                lambda x: self.get_symbol_from_file(x).upper() not in self._old_instruments,
                self.df_files,
            )
        )
        with tqdm(total=len(new_stock_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as execute:
                for file_path, (_begin_time, _end_time) in zip(new_stock_files, execute.map(_fun, new_stock_files)):
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(_end_time, pd.Timestamp):
                        symbol = self.get_symbol_from_file(file_path).upper()
                        _dt_map = self._old_instruments.setdefault(symbol, dict())
                        _dt_map[self.INSTRUMENTS_START_FIELD] = self._format_datetime(_begin_time)
                        _dt_map[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end_time)
                    p_bar.update()
        _inst_df = pd.DataFrame.from_dict(self._old_instruments, orient="index")
        _inst_df.index.names = [self.symbol_field_name]
        self.save_instruments(_inst_df.reset_index())
        logger.info("end of instruments dump.\n")

    def dump(self):
        self._calendars_list = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))
        # noinspection PyAttributeOutsideInit
        self._old_instruments = (
            self._read_instruments(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME))
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )  # type: dict
        self._dump_instruments()
        self._dump_features()


class DumpDataUpdate(DumpDataBase):
    def __init__(
        self,
        data_path: str,
        quant_master_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        data_path: str
            stock data path or directory
        quant_master_dir: str
            quant_master(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup quant_master_dir to backup_dir
        freq: str, default "day"
            transaction frequency
        max_workers: int, default None
            number of threads
        date_field_name: str, default "date"
            the name of the date field in the csv
        file_suffix: str, default ".csv"
            file suffix
        symbol_field_name: str, default "symbol"
            symbol field name
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        """
        super().__init__(
            data_path,
            quant_master_dir,
            backup_dir,
            freq,
            max_workers,
            date_field_name,
            file_suffix,
            symbol_field_name,
            exclude_fields,
            include_fields,
        )
        self._mode = self.UPDATE_MODE
        self._old_calendar_list = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))
        # NOTE: all.txt only exists once for each stock
        # NOTE: if a stock corresponds to multiple different time ranges, user need to modify self._update_instruments
        self._update_instruments = (
            self._read_instruments(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME))
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )  # type: dict

        # load all csv files
        self._all_data = self._load_all_source_data()  # type: pd.DataFrame
        self._new_calendar_list = self._old_calendar_list + sorted(
            filter(lambda x: x > self._old_calendar_list[-1], self._all_data[self.date_field_name].unique())
        )

    def _load_all_source_data(self):
        # NOTE: Need more memory
        logger.info("start load all source data....")
        all_df = []

        def _read_df(file_path: Path):
            _df = read_as_df(file_path)
            if self.date_field_name in _df.columns and not np.issubdtype(
                _df[self.date_field_name].dtype, np.datetime64
            ):
                _df[self.date_field_name] = pd.to_datetime(_df[self.date_field_name])
            if self.symbol_field_name not in _df.columns:
                _df[self.symbol_field_name] = self.get_symbol_from_file(file_path)
            return _df

        with tqdm(total=len(self.df_files)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as executor:
                for df in executor.map(_read_df, self.df_files):
                    if not df.empty:
                        all_df.append(df)
                    p_bar.update()

        logger.info("end of load all data.\n")
        return pd.concat(all_df, sort=False)

    def _dump_calendars(self):
        pass

    def _dump_instruments(self):
        pass

    def _dump_features(self):
        logger.info("start dump features......")
        error_code = {}
        with ProcessPoolExecutor(max_workers=self.works) as executor:
            futures = {}
            for _code, _df in self._all_data.groupby(self.symbol_field_name, group_keys=False):
                _code = fname_to_code(str(_code).lower()).upper()
                _start, _end = self._get_date(_df, is_begin_end=True)
                if not (isinstance(_start, pd.Timestamp) and isinstance(_end, pd.Timestamp)):
                    continue
                if _code in self._update_instruments:
                    # exists stock, will append data
                    _update_calendars = (
                        _df[_df[self.date_field_name] > self._update_instruments[_code][self.INSTRUMENTS_END_FIELD]][
                            self.date_field_name
                        ]
                        .sort_values()
                        .to_list()
                    )
                    if _update_calendars:
                        _existing_end = self._update_instruments[_code][self.INSTRUMENTS_END_FIELD]
                        self._update_instruments[_code][self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                        _new_df = _df[_df[self.date_field_name] > _existing_end]
                        futures[executor.submit(self._dump_bin, _new_df, _update_calendars)] = _code
                else:
                    # new stock
                    _dt_range = self._update_instruments.setdefault(_code, dict())
                    _dt_range[self.INSTRUMENTS_START_FIELD] = self._format_datetime(_start)
                    _dt_range[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                    futures[executor.submit(self._dump_bin, _df, self._new_calendar_list)] = _code

            with tqdm(total=len(futures)) as p_bar:
                for _future in as_completed(futures):
                    try:
                        _future.result()
                    except Exception:
                        error_code[futures[_future]] = traceback.format_exc()
                    p_bar.update()
            logger.info(f"dump bin errors: {error_code}")

        logger.info("end of features dump.\n")

    def _cleanup_stale_instruments(self):
        """Remove instruments whose end_date is more than 180 days before the last calendar date.

        Saves removed entries to instruments/all.txt.bak.YYYYMMDD for recovery.
        Also deletes the corresponding features/<symbol>/ directories.
        """
        cal_path = self._calendars_dir.joinpath(f"{self.freq}.txt")
        if not cal_path.exists():
            return
        cal_lines = cal_path.read_text().strip().split("\n")
        if not cal_lines:
            return
        last_cal_date = pd.Timestamp(cal_lines[-1].strip())
        cutoff = last_cal_date - pd.Timedelta(days=180)

        removed = {}
        for _code, _info in list(self._update_instruments.items()):
            end_dt = pd.Timestamp(_info[self.INSTRUMENTS_END_FIELD])
            if end_dt < cutoff:
                removed[_code] = self._update_instruments.pop(_code)

        if removed:
            # Save backup of removed entries
            bak_path = self._instruments_dir.joinpath(
                f"all.txt.bak.{pd.Timestamp.now().strftime('%Y%m%d')}"
            )
            lines = [
                f"{code}\t{info[self.INSTRUMENTS_START_FIELD]}\t{info[self.INSTRUMENTS_END_FIELD]}"
                for code, info in removed.items()
            ]
            bak_path.write_text("\n".join(lines) + "\n")

            # Remove orphaned feature directories
            removed_dirs = 0
            for _code in removed:
                feat_dir = self._features_dir / code_to_fname(_code).lower()
                if feat_dir.exists():
                    shutil.rmtree(feat_dir)
                    removed_dirs += 1

            logger.info(
                f"Removed {len(removed)} stale instruments "
                f"({removed_dirs} feature dirs deleted, backup: {bak_path.name})."
            )
        else:
            logger.info("No stale instruments found.")

    def dump(self):
        self.save_calendars(self._new_calendar_list)
        self._dump_features()
        self._cleanup_stale_instruments()
        df = pd.DataFrame.from_dict(self._update_instruments, orient="index")
        df.index.names = [self.symbol_field_name]
        self.save_instruments(df.reset_index())


def verify_dump(quant_master_dir: str, expected_end_date: str = None) -> bool:
    """Verify quant_master data integrity after a dump operation.

    Parameters
    ----------
    quant_master_dir : str
        Path to the quant_master data directory.
    expected_end_date : str, optional
        If given, verify the last calendar date matches this date.

    Returns
    -------
    bool
        True if all checks pass.
    """
    import random

    qm_dir = Path(quant_master_dir).expanduser().resolve()
    all_pass = True

    # 1. Calendar check
    cal_path = qm_dir / "calendars" / "day.txt"
    if not cal_path.exists():
        logger.error("verify: calendars/day.txt does not exist")
        return False
    cal_lines = cal_path.read_text().strip().split("\n")
    last_cal_date = cal_lines[-1].strip() if cal_lines else ""
    if expected_end_date and last_cal_date != expected_end_date:
        logger.error(f"verify: last calendar date {last_cal_date} != expected {expected_end_date}")
        all_pass = False
    else:
        logger.info(f"verify: calendar OK (last date: {last_cal_date})")

    # 2. Instruments staleness check
    inst_path = qm_dir / "instruments" / "all.txt"
    if not inst_path.exists():
        logger.error("verify: instruments/all.txt does not exist")
        return False
    last_cal_ts = pd.Timestamp(last_cal_date)
    stale_cutoff = last_cal_ts - pd.Timedelta(days=365)
    stale_count = 0
    total_count = 0
    with open(inst_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            total_count += 1
            end_dt = pd.Timestamp(parts[2])
            if end_dt < stale_cutoff:
                stale_count += 1
    if stale_count > 0:
        logger.warning(f"verify: {stale_count}/{total_count} instruments have end_date > 1 year old")
    else:
        logger.info(f"verify: instruments OK ({total_count} entries, no stale)")

    # 3. Spot-check feature dirs
    features_dir = qm_dir / "features"
    if features_dir.exists():
        all_dirs = [d for d in features_dir.iterdir() if d.is_dir()]
        sample_size = min(10, len(all_dirs))
        sample_dirs = random.sample(all_dirs, sample_size) if all_dirs else []
        bad_dirs = 0
        for d in sample_dirs:
            bin_files = list(d.glob("*.day.bin"))
            if not bin_files:
                logger.warning(f"verify: {d.name} has no .day.bin files")
                bad_dirs += 1
                continue
            # Check first bin file for all-NaN
            try:
                data = np.fromfile(str(bin_files[0]), dtype="<f4")
                if len(data) < 2:
                    logger.warning(f"verify: {d.name}/{bin_files[0].name} too short ({len(data)} values)")
                    bad_dirs += 1
                elif np.all(np.isnan(data[1:])):
                    logger.warning(f"verify: {d.name}/{bin_files[0].name} is all NaN")
                    bad_dirs += 1
            except Exception as e:
                logger.warning(f"verify: {d.name}/{bin_files[0].name} read error: {e}")
                bad_dirs += 1
        if bad_dirs > 0:
            logger.warning(f"verify: {bad_dirs}/{sample_size} sampled feature dirs have issues")
            all_pass = False
        else:
            logger.info(f"verify: feature dirs OK ({sample_size} sampled, all good)")
    else:
        logger.warning("verify: features directory does not exist")

    if all_pass:
        logger.info("verify: ALL CHECKS PASSED")
    else:
        logger.warning("verify: SOME CHECKS FAILED — review warnings above")
    return all_pass


if __name__ == "__main__":
    fire.Fire({"dump_all": DumpDataAll, "dump_fix": DumpDataFix, "dump_update": DumpDataUpdate, "verify_dump": verify_dump})
