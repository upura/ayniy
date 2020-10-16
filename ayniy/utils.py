import datetime
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 777) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def timer(name: str) -> Any:
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


class Data:
    @classmethod
    def dump(cls, value: Any, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path: str) -> Any:
        return joblib.load(path)


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


class Logger:
    def __init__(self) -> None:
        self.general_logger = logging.getLogger("general")
        self.result_logger = logging.getLogger("result")
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        file_general_handler = logging.FileHandler("../output/logs/general.log")
        file_result_handler = logging.FileHandler("../output/logs/result.log")
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message: str) -> None:
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    def result(self, message: str) -> None:
        self.result_logger.info(message)

    def result_ltsv(self, dic: Dict) -> None:
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name: str, scores: List) -> None:
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic["name"] = run_name
        dic["score"] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f"score{i}"] = score
        self.result(self.to_ltsv(dic))

    def now_string(self) -> str:
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_ltsv(self, dic: Dict) -> str:
        return "\t".join(["{}:{}".format(key, value) for key, value in dic.items()])


class FeatureStore:
    def __init__(self, feature_names: str, target_col: str) -> None:
        self.feature_names = feature_names
        self.target_col = target_col
        _res = pd.concat([pd.read_feather(f) for f in feature_names], axis=1)

        _train = _res.dropna(subset=[target_col]).copy()
        _test = _res.loc[_res[target_col].isnull()].copy()

        self.X_train = _train.drop(target_col, axis=1)
        self.y_train = _train[target_col]
        self.X_test = _test.drop(target_col, axis=1)
