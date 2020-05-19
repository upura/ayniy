import numpy as np
from sklearn.model_selection import StratifiedKFold

from ayniy.preprocessing import (count_null,
                                 label_encoding,
                                 frequency_encoding,
                                 count_encoding,
                                 count_encoding_interact,
                                 matrix_factorization,
                                 target_encoding,
                                 aggregation,
                                 numeric_interact,
                                 delete_cols,
                                 detect_delete_cols,
                                 save_as_pickle,
                                 get_tfidf,
                                 get_count,
                                 get_bert,
                                 get_swem_mean,
                                 use_cols)


def test_use_cols(load_titanic):
    train, test = load_titanic
    encode_col = ['embarked', 'sex']
    target_col = 'survived'
    train, test = use_cols(train, test, {'encode_col': encode_col, 'target_col': target_col})
    assert train.shape[1] == len(encode_col) + 1
    assert list(set(train.columns) - set(test.columns)) == [target_col]


def test_delete_cols(load_titanic):
    train, test = load_titanic
    encode_col = ['embarked']
    train, test = delete_cols(train, test, {'encode_col': encode_col})
    assert 'embarked' not in train.columns


def test_count_null(load_titanic):
    train, test = load_titanic
    encode_col = ['embarked', 'sex']
    train, test = count_null(train, test, {'encode_col': encode_col})
    assert 'count_null' in train.columns


def test_label_encoding(load_titanic):
    train, test = load_titanic
    encode_col = ['embarked', 'sex']
    train, _ = label_encoding(train, test, {'encode_col': encode_col})
    assert train[encode_col[0]].dtype == np.int64


def test_label_encoding_unseen(load_titanic):
    train, test = load_titanic
    encode_col = ['NOT_IN_COLUMNS']
    train_after, _ = label_encoding(train, test, {'encode_col': encode_col})
    assert train.shape == train_after.shape


def test_frequency_encoding(load_titanic):
    train, test = load_titanic
    encode_col = ['embarked', 'sex']
    prefix = 'fe_'
    train, _ = frequency_encoding(train, test, {'encode_col': encode_col})
    assert train[prefix + encode_col[0]].dtype == np.float64


def test_count_encoding(load_titanic):
    train, test = load_titanic
    encode_col = ['embarked', 'sex']
    prefix = 'ce_'
    train, _ = count_encoding(train, test, {'encode_col': encode_col})
    assert train[prefix + encode_col[0]].dtype == np.float64


def test_count_encoding_interact(load_titanic):
    train, test = load_titanic
    encode_col = ['embarked', 'sex']
    prefix = 'cei_'
    train, _ = count_encoding_interact(train, test, {'encode_col': encode_col})
    assert train[f'{prefix}{encode_col[0]}_{encode_col[1]}'].dtype == np.int64


def test_numeric_interact(load_titanic):
    train, test = load_titanic
    encode_col = ['age', 'fare']
    cols = ['_plus_', '_mul_', '_div_']
    train, _ = numeric_interact(train, test, {'encode_col': encode_col})
    for c in cols:
        assert train[f'{encode_col[0]}{c}{encode_col[1]}'].dtype == np.float64


def test_target_encoding(load_titanic):
    train, test = load_titanic
    encode_col = ['embarked', 'sex']
    target_col = 'survived'
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    prefix = 'te_'
    train, _ = target_encoding(train, test,
                               {'encode_col': encode_col, 'target_col': target_col},
                               {'cv': cv})
    assert train[prefix + encode_col[0]].dtype == np.float64
