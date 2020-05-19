import os
import datetime

import pandas as pd
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
                                 standerize,
                                 fillna,
                                 datatime_parser,
                                 circle_encoding,
                                 use_cols,
                                 delete_cols,
                                 detect_delete_cols,
                                 save_as_pickle)


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


def test_standerize(load_titanic):
    train, test = load_titanic
    encode_col = ['age', 'fare']
    train, _ = standerize(train, test, {'encode_col': encode_col})
    SMALL_ENOUGH = 0.000001
    assert np.mean(train['age']) < SMALL_ENOUGH
    assert 1 - np.std(train['age']) < SMALL_ENOUGH


def test_fillna(load_titanic):
    train, test = load_titanic
    train, _ = fillna(train, test,
                      {'encode_col': ['age']},
                      {'how': 'median'})
    train, _ = fillna(train, test,
                      {'encode_col': ['fare']},
                      {'how': 'mean'})
    assert train['age'].isnull().sum() == 0
    assert train['fare'].isnull().sum() == 0


def test_datatime_parser(load_titanic):
    train, test = load_titanic
    train['now'] = datetime.datetime.now()
    test['now'] = datetime.datetime.now()
    train_new, _ = datatime_parser(train, test, {'encode_col': ['now']})
    assert len((set(train_new.columns) - set(train.columns))) > 0


def test_circle_encoding(load_titanic):
    train = pd.DataFrame({'numbers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
    test = train.copy()
    train_new, _ = circle_encoding(train, test, {'encode_col': ['numbers']})
    assert len((set(train_new.columns) - set(train.columns))) == 2


def test_matrix_factorization(load_titanic):
    train, test = load_titanic
    encode_col = ['pclass', 'sibsp']
    n_components_lda = 3
    n_components_svd = 4
    train_new, _ = matrix_factorization(train, test,
                                        {'encode_col': encode_col},
                                        {'n_components_lda': n_components_lda, 'n_components_svd': n_components_svd})
    assert len((set(train_new.columns) - set(train.columns))) > 0


def test_aggregation(load_titanic):
    train, test = load_titanic
    groupby_dict = [
        {
            'key': ['pclass'],
            'var': ['age', 'fare'],
            'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
        },
    ]
    nunique_dict = [
        {
            'key': ['pclass'],
            'var': ['sibsp'],
            'agg': ['nunique']
        },
    ]
    train_new, _ = aggregation(train, test,
                               {'groupby_dict': groupby_dict, 'nunique_dict': nunique_dict})
    assert len((set(train_new.columns) - set(train.columns))) > 0


def test_detect_delete_cols(load_titanic):
    train, test = load_titanic
    escape_col = ['sex', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone']
    unique_cols, duplicated_cols, high_corr_cols = detect_delete_cols(
        train, test, {'escape_col': escape_col}, {'threshold': 0.1})
    assert type(unique_cols) == list
    assert type(duplicated_cols) == list
    assert type(high_corr_cols) == list


def test_save_as_pickle(load_titanic):
    train, test = load_titanic
    exp_id = 'pytest'
    output_dir = 'input'
    save_as_pickle(train, test,
                   {'target_col': 'survived'},
                   {'output_dir': output_dir, 'exp_id': exp_id})
    assert os.path.exists(f'{output_dir}/X_train_{exp_id}.pkl')
    assert os.path.exists(f'{output_dir}/y_train_{exp_id}.pkl')
    assert os.path.exists(f'{output_dir}/X_test_{exp_id}.pkl')
    os.remove(f'{output_dir}/X_train_{exp_id}.pkl')
    os.remove(f'{output_dir}/y_train_{exp_id}.pkl')
    os.remove(f'{output_dir}/X_test_{exp_id}.pkl')


def test_save_as_pickle_test_exclude_target(load_titanic):
    train, test = load_titanic
    test.drop('survived', axis=1, inplace=True)
    exp_id = 'pytest'
    output_dir = 'input'
    save_as_pickle(train, test,
                   {'target_col': 'survived'},
                   {'output_dir': output_dir, 'exp_id': exp_id})
    assert os.path.exists(f'{output_dir}/X_train_{exp_id}.pkl')
    assert os.path.exists(f'{output_dir}/y_train_{exp_id}.pkl')
    assert os.path.exists(f'{output_dir}/X_test_{exp_id}.pkl')
    os.remove(f'{output_dir}/X_train_{exp_id}.pkl')
    os.remove(f'{output_dir}/y_train_{exp_id}.pkl')
    os.remove(f'{output_dir}/X_test_{exp_id}.pkl')
