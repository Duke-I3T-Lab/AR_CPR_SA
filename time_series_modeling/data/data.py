from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
from itertools import repeat, chain

import numpy as np
import pandas as pd
from tqdm import tqdm

# from sktime.utils import load_data

from datasets import utils
from functools import partial

logger = logging.getLogger("__main__")


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both")
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):
    def set_num_processes(self, n_proc):
        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class EyeTrackingData(BaseData):
    """
    Dataset class for Eye Tracking dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(
        self,
        root_dir=None,
        file_list=None,
        pattern=None,
        n_proc=1,
        limit_size=None,
        config=None,
        user_ids=None,
        clean=False,
    ):
        self.set_num_processes(n_proc=n_proc)

        self.all_df, self.labels_df = self.load_all(
            root_dir,
            file_list=file_list,
            pattern=pattern,
            user_ids=user_ids,
            clean=clean,
        )

        # self.all_df = self.all_df.sort_values(
        #     by=["machine_record_index"]
        # )  # datasets is presorted
        # self.all_df = self.all_df.set_index("machine_record_index")

        self.all_df = self.all_df.set_index("ts_index")
        self.labels_df = self.labels_df.set_index("ts_index")
        label_num = len(self.labels_df.columns)
        self.class_names = list(range(label_num))

        # count 1's at each column

        for class_name in self.class_names:
            print(
                "number of samples for class {}: {}".format(
                    class_name,
                    len(self.labels_df.loc[self.labels_df[str(class_name)] == 1]),
                )
            )

        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs
        self.max_seq_len = 150
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df[self.feature_names]

    def load_all(
        self, root_dir, file_list=None, pattern=None, user_ids=None, clean=False
    ):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # each file name corresponds to another date. Also tools (A, B) and others.

        # Select paths for training and evaluation

        if file_list is None and user_ids is None:
            # find all files at arbitrary depth
            data_paths = []
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(".csv"):
                        data_paths.append(os.path.join(root, file))
        elif user_ids is not None:
            data_paths = []
            for user_id in user_ids:
                data_paths += glob.glob(os.path.join(root_dir, str(user_id), "*.csv"))

        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception(
                "No files found using: {}".format(os.path.join(root_dir, "*"))
            )

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [
            p for p in selected_paths if os.path.isfile(p) and p.endswith(".csv")
        ]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))
        feature_paths = [
            p for p in input_paths if "feature_df" in p and "label_df" not in p
        ]
        label_paths = [
            p for p in input_paths if "label_df" in p and "feature_df" not in p
        ]

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(
                self.n_proc, len(input_paths)
            )  # no more than file_names needed here
            logger.info(
                "Loading {} datasets files using {} parallel processes ...".format(
                    len(input_paths), _n_proc
                )
            )
            with Pool(processes=_n_proc) as pool:
                all_feature_df = pd.concat(
                    pool.map(EyeTrackingData.load_single, feature_paths)
                )
                all_label_df = pd.concat(
                    pool.map(
                        partial(EyeTrackingData.load_single, is_label=True), label_paths
                    )
                )

        else:  # read 1 file at a time
            all_feature_df = pd.concat(
                EyeTrackingData.load_single(path, clean=clean, load_index=i)
                for i, path in enumerate(feature_paths)
            )

            all_label_df = pd.concat(
                EyeTrackingData.load_single(
                    path, is_label=True, clean=clean, load_index=i
                )
                for i, path in enumerate(label_paths)
            )

        return all_feature_df, all_label_df

    @staticmethod
    def load_single(filepath, is_label=False, clean=False, load_index=0):
        df = EyeTrackingData.read_data(filepath, is_label=is_label)
        if (
            clean
        ):  # files are being reused in some cases, so we need to reset the index to avoid duplicates
            # add load_index * 10000 to ts_index
            df["ts_index"] = df["ts_index"] + load_index * 10000
        df = EyeTrackingData.select_columns(df)
        # check if there are any nan values
        if is_label:
            if df.isna().sum().sum() > 0:
                print(filepath)
                raise ValueError("NaN values in label dataframe")

        if is_label:
            return df
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning(
                "{} nan values in {} will be replaced by 0".format(num_nan, filepath)
            )
            df = df.fillna(0)

        return df

    @staticmethod
    def read_data(filepath, is_label=False):
        """Reads a single .csv, which typically contains a day of datasets of various machine sessions."""
        df = pd.read_csv(filepath, index_col=False)

        return df

    @staticmethod
    def select_columns(df):
        return df


data_factory = {
    "ar": EyeTrackingData,
}
