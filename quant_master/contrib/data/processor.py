import numpy as np

from ...log import TimeInspector
from ...data.dataset.processor import Processor, _assign_columns, get_group_columns


class ConfigSectionProcessor(Processor):
    """
    This processor is designed for Alpha158. And will be replaced by simple processors in the future
    """

    def __init__(self, fields_group=None, **kwargs):
        super().__init__()
        # Options
        self.fillna_feature = kwargs.get("fillna_feature", True)
        self.fillna_label = kwargs.get("fillna_label", True)
        self.clip_feature_outlier = kwargs.get("clip_feature_outlier", False)
        self.shrink_feature_outlier = kwargs.get("shrink_feature_outlier", True)
        self.clip_label_outlier = kwargs.get("clip_label_outlier", False)

        self.fields_group = None

    def __call__(self, df):
        return self._transform(df)

    def _transform(self, df):
        TimeInspector.set_time_mark()

        # Copy the focus part and change it to single level
        selected_cols = get_group_columns(df, self.fields_group)
        df_focus = df[selected_cols].copy()
        if len(df_focus.columns.levels) > 1:
            df_focus = df_focus.droplevel(level=0)

        label_cols = [c for c in df_focus.columns if c.startswith("LABEL")]

        # Identify column groups
        # Note: ^KLEN|^KLOW|^KUP also matches KLOW2/KUP2 columns, which is
        # the same behavior as the original code.
        # Note: ^MA in standard also matches MAX5, which is the same behavior
        # as the original code (MAX5 is processed in both standard and MAX groups).

        _cols = [
            "KMID", "KSFT", "OPEN", "HIGH", "LOW", "CLOSE", "VWAP", "ROC", "MA",
            "BETA", "RESI", "QTLU", "QTLD", "RSV", "SUMP", "SUMN", "SUMD",
            "VSUMP", "VSUMN", "VSUMD",
        ]
        pat = "|".join(["^" + x for x in _cols])
        standard_cols = list(df_focus.columns[
            df_focus.columns.str.contains(pat) & (~df_focus.columns.isin(["HIGH0", "LOW0"]))
        ])

        log_cols = list(df_focus.columns[df_focus.columns.str.contains("^STD|^VOLUME|^VMA|^VSTD")])
        fillna_cols = list(df_focus.columns[df_focus.columns.str.contains("^RSQR")])
        max_cols = list(df_focus.columns[df_focus.columns.str.contains("^MAX|^HIGH0")])
        min_cols = list(df_focus.columns[df_focus.columns.str.contains("^MIN|^LOW0")])
        exp_cols = list(df_focus.columns[df_focus.columns.str.contains("^CORR|^CORD")])
        log1p_cols = list(df_focus.columns[df_focus.columns.str.contains("^WVMA")])
        klen_cols = list(df_focus.columns[df_focus.columns.str.contains("^KLEN|^KLOW|^KUP")])
        klow2_cols = list(df_focus.columns[df_focus.columns.str.contains("^KLOW2|^KUP2")])

        # ---- Pre-transform columns that don't overlap with standard ----
        # (standard/MAX must be separate because MAX5 is in both; MAX pre-transform
        #  must happen AFTER standard normalization to match original behavior)
        for c in log_cols:
            df_focus[c] = np.log(df_focus[c])
        for c in fillna_cols:
            df_focus[c] = df_focus[c].fillna(0)
        for c in min_cols:
            df_focus[c] = (1 - df_focus[c]) ** 0.5
        for c in exp_cols:
            df_focus[c] = np.exp(df_focus[c])
        for c in log1p_cols:
            df_focus[c] = np.log1p(df_focus[c])
        for c in klen_cols:
            df_focus[c] = df_focus[c] ** 0.25

        # Phase 1 groupby: labels + standard + log + fillna + min + exp + log1p + KLEN
        # (standard has NO pre-transform; MAX/HIGH0 excluded)
        phase1_cols = list(dict.fromkeys(
            standard_cols + log_cols + fillna_cols + min_cols + exp_cols + log1p_cols + klen_cols
        ))

        def _normalize_phase1(g):
            for c in label_cols:
                col = g[c] - g[c].mean()
                col = col / g[c].std()
                if self.clip_label_outlier:
                    col = col.clip(-3, 3)
                if self.fillna_label:
                    col = col.fillna(0)
                g[c] = col

            for c in phase1_cols:
                col = g[c] - g[c].median()
                col = col / (col.abs().median() * 1.4826)
                if self.clip_feature_outlier:
                    col = col.clip(-3, 3)
                if self.shrink_feature_outlier:
                    col = col.where(col <= 3, 3 + (col - 3).div(col.max() - 3) * 0.5)
                    col = col.where(col >= -3, -3 - (col + 3).div(col.min() + 3) * 0.5)
                if self.fillna_feature:
                    col = col.fillna(0)
                g[c] = col

            return g

        df_focus = df_focus.groupby(level="datetime", group_keys=False).apply(_normalize_phase1)
        if not df_focus.index.equals(df.index):
            raise ValueError(
                f"{self.__class__.__name__} phase1 changed index: input_len={len(df)}, output_len={len(df_focus)}"
            )

        # Phase 2 groupby: MAX/HIGH0 (pre-transform applied AFTER standard normalization)
        for c in max_cols:
            df_focus[c] = (df_focus[c] - 1) ** 0.5

        if max_cols:

            def _normalize_max(g):
                for c in max_cols:
                    col = g[c] - g[c].median()
                    col = col / (col.abs().median() * 1.4826)
                    if self.clip_feature_outlier:
                        col = col.clip(-3, 3)
                    if self.shrink_feature_outlier:
                        col = col.where(col <= 3, 3 + (col - 3).div(col.max() - 3) * 0.5)
                        col = col.where(col >= -3, -3 - (col + 3).div(col.min() + 3) * 0.5)
                    if self.fillna_feature:
                        col = col.fillna(0)
                    g[c] = col

                return g

            df_focus = df_focus.groupby(level="datetime", group_keys=False).apply(_normalize_max)
            if not df_focus.index.equals(df.index):
                raise ValueError(
                    f"{self.__class__.__name__} max-phase changed index: input_len={len(df)}, output_len={len(df_focus)}"
                )

        # Phase 3 groupby: KLOW2/KUP2 (pre-transform on already-normalized KLEN values)
        for c in klow2_cols:
            df_focus[c] = df_focus[c] ** 0.5

        if klow2_cols:

            def _normalize_klow2(g):
                for c in klow2_cols:
                    col = g[c] - g[c].median()
                    col = col / (col.abs().median() * 1.4826)
                    if self.clip_feature_outlier:
                        col = col.clip(-3, 3)
                    if self.shrink_feature_outlier:
                        col = col.where(col <= 3, 3 + (col - 3).div(col.max() - 3) * 0.5)
                        col = col.where(col >= -3, -3 - (col + 3).div(col.min() + 3) * 0.5)
                    if self.fillna_feature:
                        col = col.fillna(0)
                    g[c] = col

                return g

            df_focus = df_focus.groupby(level="datetime", group_keys=False).apply(_normalize_klow2)
            if not df_focus.index.equals(df.index):
                raise ValueError(
                    f"{self.__class__.__name__} klow2-phase changed index: input_len={len(df)}, output_len={len(df_focus)}"
                )

        return _assign_columns(df, selected_cols, df_focus, self.__class__.__name__)

        TimeInspector.log_cost_time("Finished preprocessing data.")

        return df
