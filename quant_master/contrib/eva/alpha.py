"""
Here is a batch of evaluation functions.

The interface should be redesigned carefully in the future.
"""

import pandas as pd
from typing import Tuple
from quant_master import get_module_logger
from quant_master.utils.paral import complex_parallel, DelayedDict
from joblib import Parallel, delayed


def calc_long_short_prec(
    pred: pd.Series, label: pd.Series, date_col="datetime", quantile: float = 0.2, dropna=False, is_alpha=False
) -> Tuple[pd.Series, pd.Series]:
    """
    calculate the precision for long and short operation


    :param pred/label: index is **pd.MultiIndex**, index name is **[datetime, instruments]**; columns names is **[score]**.

            .. code-block:: python
                                                  score
                datetime            instrument
                2020-12-01 09:30:00 SH600068    0.553634
                                    SH600195    0.550017
                                    SH600276    0.540321
                                    SH600584    0.517297
                                    SH600715    0.544674
    label :
        label
    date_col :
        date_col

    Returns
    -------
    (pd.Series, pd.Series)
        long precision and short precision in time level
    """
    if is_alpha:
        label = label - label.groupby(level=date_col, group_keys=False).mean()
    if int(1 / quantile) >= len(label.index.get_level_values(1).unique()):
        raise ValueError("Need more instruments to calculate precision")

    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)

    # Vectorized quantile-based masking: rank predictions within each date group,
    # then select top/bottom N = int(len(group) * quantile) items.
    # Descending rank for long (matches nlargest), ascending rank for short (matches nsmallest).
    group_col = df.groupby(level=date_col, group_keys=False)["pred"]
    df["_n"] = group_col.transform("count")
    df["_N"] = (df["_n"] * quantile).astype(int)
    df["_rank_desc"] = group_col.rank(method="first", ascending=False)
    df["_rank_asc"] = group_col.rank(method="first", ascending=True)
    long_mask = df["_rank_desc"] <= df["_N"]
    short_mask = df["_rank_asc"] <= df["_N"]

    long_pos = (df.loc[long_mask, "label"] > 0).groupby(level=date_col).sum()
    long_cnt = long_mask.groupby(level=date_col).sum()
    short_neg = (df.loc[short_mask, "label"] < 0).groupby(level=date_col).sum()
    short_cnt = short_mask.groupby(level=date_col).sum()

    return long_pos / long_cnt, short_neg / short_cnt


def calc_long_short_return(
    pred: pd.Series,
    label: pd.Series,
    date_col: str = "datetime",
    quantile: float = 0.2,
    dropna: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    calculate long-short return

    Note:
        `label` must be raw stock returns.

    Parameters
    ----------
    pred : pd.Series
        stock predictions
    label : pd.Series
        stock returns
    date_col : str
        datetime index name
    quantile : float
        long-short quantile

    Returns
    ----------
    long_short_r : pd.Series
        daily long-short returns
    long_avg_r : pd.Series
        daily long-average returns
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)
    group = df.groupby(level=date_col, group_keys=False)

    def N(x):
        return int(len(x) * quantile)

    r_long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label.mean())
    r_short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label.mean())
    r_avg = group.label.mean()
    return (r_long - r_short) / 2, r_avg


def pred_autocorr(pred: pd.Series, lag=1, inst_col="instrument", date_col="datetime"):
    """pred_autocorr.

    Limitation:
    - If the datetime is not sequential densely, the correlation will be calulated based on adjacent dates. (some users may expected NaN)

    :param pred: pd.Series with following format
                instrument  datetime
                SH600000    2016-01-04   -0.000403
                            2016-01-05   -0.000753
                            2016-01-06   -0.021801
                            2016-01-07   -0.065230
                            2016-01-08   -0.062465
    :type pred: pd.Series
    :param lag:
    """
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
        get_module_logger("pred_autocorr").warning(f"Only the first column of `pred` is kept")
    pred_ustk = pred.sort_index().unstack(inst_col)
    # Vectorized: compute cross-sectional correlation of each date's predictions
    # with those from `lag` dates earlier, across all instruments at once.
    corr_s = pred_ustk.corrwith(pred_ustk.shift(lag), axis=1)
    corr_s.name = None
    return corr_s.sort_index()


def pred_autocorr_all(pred_dict, n_jobs=-1, **kwargs):
    """
    calculate auto correlation for pred_dict

    Parameters
    ----------
    pred_dict : dict
        A dict like {<method_name>:  <prediction>}
    kwargs :
        all these arguments will be passed into pred_autocorr
    """
    ac_dict = {}
    for k, pred in pred_dict.items():
        ac_dict[k] = delayed(pred_autocorr)(pred, **kwargs)
    return complex_parallel(Parallel(n_jobs=n_jobs, verbose=10), ac_dict)


def calc_ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False) -> (pd.Series, pd.Series):
    """calc_ic.

    Parameters
    ----------
    pred :
        pred
    label :
        label
    date_col :
        date_col

    Returns
    -------
    (pd.Series, pd.Series)
        ic and rank ic
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    # Fused: compute both Pearson IC and Spearman rank IC in a single groupby pass.
    def _ic(group):
        return group["pred"].corr(group["label"]), group["pred"].corr(group["label"], method="spearman")

    ic_ric = df.groupby(date_col, group_keys=False).apply(_ic)
    ic = ic_ric.apply(lambda x: x[0])
    ric = ic_ric.apply(lambda x: x[1])
    if dropna:
        return ic.dropna(), ric.dropna()
    else:
        return ic, ric


def calc_all_ic(pred_dict_all, label, date_col="datetime", dropna=False, n_jobs=-1):
    """calc_all_ic.

    Parameters
    ----------
    pred_dict_all :
        A dict like {<method_name>:  <prediction>}
    label:
        A pd.Series of label values

    Returns
    -------
    {'Q2+IND_z': {'ic': <ic series like>
                          2016-01-04   -0.057407
                          ...
                          2020-05-28    0.183470
                          2020-05-29    0.171393
                  'ric': <rank ic series like>
                          2016-01-04   -0.040888
                          ...
                          2020-05-28    0.236665
                          2020-05-29    0.183886
                  }
    ...}
    """
    pred_all_ics = {}
    for k, pred in pred_dict_all.items():
        pred_all_ics[k] = DelayedDict(["ic", "ric"], delayed(calc_ic)(pred, label, date_col=date_col, dropna=dropna))
    pred_all_ics = complex_parallel(Parallel(n_jobs=n_jobs, verbose=10), pred_all_ics)
    return pred_all_ics
