import datetime

import numpy as np
import pandas as pd

from ayniy.preprocessing import (
    aggregation,
    circle_encoding,
    count_null,
    datetime_parser,
    detect_delete_cols,
    fillna,
    frequency_encoding,
    matrix_factorization,
    standerize,
)


def test_count_null(load_titanic):
    train, test = load_titanic
    encode_col = ["embarked", "sex"]
    train, test = count_null(train, test, encode_col)
    assert "count_null" in train.columns


def test_frequency_encoding(load_titanic):
    train, test = load_titanic
    encode_col = ["embarked", "sex"]
    prefix = "fe_"
    train, _ = frequency_encoding(train, test, encode_col)
    assert train[prefix + encode_col[0]].dtype == np.float64


def test_standerize(load_titanic):
    train, test = load_titanic
    encode_col = ["age", "fare"]
    train, _ = standerize(train, test, encode_col)
    SMALL_ENOUGH = 0.000001
    assert np.mean(train["age"]) < SMALL_ENOUGH
    assert 1 - np.std(train["age"]) < SMALL_ENOUGH


def test_fillna(load_titanic):
    train, test = load_titanic
    train, _ = fillna(train, test, encode_col=["age"], how="median")
    train, _ = fillna(train, test, encode_col=["fare"], how="mean")
    assert train["age"].isnull().sum() == 0
    assert train["fare"].isnull().sum() == 0


def test_datetime_parser(load_titanic):
    train, test = load_titanic
    train["now"] = datetime.datetime.now()
    test["now"] = datetime.datetime.now()
    encode_col = ["now"]
    train_new, _ = datetime_parser(train, test, encode_col)
    assert len((set(train_new.columns) - set(train.columns))) > 0


def test_circle_encoding(load_titanic):
    train = pd.DataFrame({"numbers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
    test = train.copy()
    encode_col = ["numbers"]
    train_new, _ = circle_encoding(train, test, encode_col)
    assert len((set(train_new.columns) - set(train.columns))) == 2


def test_matrix_factorization(load_titanic):
    train, test = load_titanic
    encode_col = ["pclass", "sibsp"]
    n_components_lda = 3
    n_components_svd = 4
    train_new, _ = matrix_factorization(
        train, test, encode_col, n_components_lda, n_components_svd
    )
    assert len((set(train_new.columns) - set(train.columns))) > 0


def test_aggregation(load_titanic):
    train, test = load_titanic
    groupby_dict = [
        {
            "key": ["pclass"],
            "var": ["age", "fare"],
            "agg": ["mean", "sum", "median", "min", "max", "var", "std"],
        },
    ]
    nunique_dict = [
        {"key": ["pclass"], "var": ["sibsp"], "agg": ["nunique"]},
    ]
    train_new, _ = aggregation(train, test, groupby_dict, nunique_dict)
    assert len((set(train_new.columns) - set(train.columns))) > 0


def test_detect_delete_cols(load_titanic):
    train, test = load_titanic
    escape_col = [
        "sex",
        "class",
        "who",
        "adult_male",
        "deck",
        "embark_town",
        "alive",
        "alone",
    ]
    threshold = 0.1
    unique_cols, duplicated_cols, high_corr_cols = detect_delete_cols(
        train, test, escape_col, threshold
    )
    assert type(unique_cols) == list
    assert type(duplicated_cols) == list
    assert type(high_corr_cols) == list
