import itertools
from os.path import join
from typing import List, Tuple
import warnings

from kaggler.preprocessing import TargetEncoder
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from ayniy.utils import Data


def use_cols(train: pd.DataFrame,
             test: pd.DataFrame,
             encode_col: List[str],
             target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select columns

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns
        target_col (str): target column

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    train = train[encode_col + [target_col]]
    test = test[encode_col]
    return train, test


def detect_delete_cols(train: pd.DataFrame,
                       test: pd.DataFrame,
                       escape_col: List[str],
                       threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect unnecessary columns for deleting

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        escape_col (List[str]): columns not encoded
        threshold (float): deleting threshold for correlations of columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    unique_cols = list(train.columns[train.nunique() == 1])
    duplicated_cols = list(train.columns[train.T.duplicated()])

    buf = train.corr()
    counter = 0
    high_corr_cols = []
    try:
        for feat_a in [x for x in train.columns if x not in escape_col]:
            for feat_b in [x for x in train.columns if x not in escape_col]:
                if feat_a != feat_b and feat_a not in high_corr_cols and feat_b not in high_corr_cols:
                    c = buf.loc[feat_a, feat_b]
                    if c > threshold:
                        counter += 1
                        high_corr_cols.append(feat_b)
                        print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
    except:
        pass
    return unique_cols, duplicated_cols, high_corr_cols


def delete_cols(train: pd.DataFrame,
                test: pd.DataFrame,
                encode_col: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Delete columns

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    train.drop(encode_col, inplace=True, axis=1)
    test.drop(encode_col, inplace=True, axis=1)
    return train, test


def label_encoding(train: pd.DataFrame,
                   test: pd.DataFrame,
                   encode_col: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Label encoding

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)
    for f in encode_col:
        try:
            lbl = preprocessing.LabelEncoder()
            train[f] = lbl.fit_transform(list(train[f].values))
        except:
            print(f)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def standerize(train: pd.DataFrame,
               test: pd.DataFrame,
               encode_col: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standerization

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    scaler = preprocessing.StandardScaler()
    train[encode_col] = scaler.fit_transform(train[encode_col])
    test[encode_col] = scaler.transform(test[encode_col])
    return train, test


def fillna(train: pd.DataFrame,
           test: pd.DataFrame,
           encode_col: List[str],
           how: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replace NaN

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns
        how (str): how to fill Nan, chosen from 'median' or 'mean'

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    for f in encode_col:
        if how == 'median':
            train[f].fillna(train[f].median(), inplace=True)
            test[f].fillna(train[f].median(), inplace=True)
        elif how == 'mean':
            train[f].fillna(train[f].mean(), inplace=True)
            test[f].fillna(train[f].mean(), inplace=True)
    return train, test


def datatime_parser(train: pd.DataFrame,
                    test: pd.DataFrame,
                    encode_col: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Datetime columns parser

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    _train = train.copy()
    _test = test.copy()
    for f in encode_col:
        _train[f + '_year'] = pd.to_datetime(train[f]).dt.year
        _train[f + '_month'] = pd.to_datetime(train[f]).dt.month
        _train[f + '_day'] = pd.to_datetime(train[f]).dt.day
        _train[f + '_dow'] = pd.to_datetime(train[f]).dt.dayofweek
        _train[f + '_hour'] = pd.to_datetime(train[f]).dt.hour
        _train[f + '_minute'] = pd.to_datetime(train[f]).dt.minute
        _test[f + '_year'] = pd.to_datetime(test[f]).dt.year
        _test[f + '_month'] = pd.to_datetime(test[f]).dt.month
        _test[f + '_day'] = pd.to_datetime(test[f]).dt.day
        _test[f + '_dow'] = pd.to_datetime(test[f]).dt.dayofweek
        _test[f + '_hour'] = pd.to_datetime(test[f]).dt.hour
        _test[f + '_minute'] = pd.to_datetime(test[f]).dt.minute
    return _train, _test


def circle_encoding(train: pd.DataFrame,
                    test: pd.DataFrame,
                    encode_col: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Circle encoding

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    _train = train.copy()
    _test = test.copy()
    for f in encode_col:
        _train[f + '_cos'] = np.cos(2 * np.pi * train[f] / train[f].max())
        _train[f + '_sin'] = np.sin(2 * np.pi * train[f] / train[f].max())
        _test[f + '_cos'] = np.cos(2 * np.pi * test[f] / train[f].max())
        _test[f + '_sin'] = np.sin(2 * np.pi * test[f] / train[f].max())
    return _train, _test


def save_as_pickle(train: pd.DataFrame,
                   test: pd.DataFrame,
                   target_col: str,
                   exp_id: str,
                   output_dir: str = '../input') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Save X_train, X_test and y_train as pickel format

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        target_col (str): target column
        exp_id (str): experiment id
        output_dir (str, optional): output directory. Defaults to '../input'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    X_train = train.drop(target_col, axis=1)
    y_train = train[target_col]
    if target_col in test.columns:
        X_test = test.drop(target_col, axis=1)
    else:
        X_test = test

    Data.dump(X_train, join(output_dir, f"X_train_{exp_id}.pkl"))
    Data.dump(y_train, join(output_dir, f"y_train_{exp_id}.pkl"))
    Data.dump(X_test, join(output_dir, f"X_test_{exp_id}.pkl"))


class GroupbyTransformer():
    def __init__(self, param_dict=None):
        self.param_dict = param_dict

    def _get_params(self, p_dict):
        key = p_dict['key']
        if 'var' in p_dict.keys():
            var = p_dict['var']
        else:
            var = self.var
        if 'agg' in p_dict.keys():
            agg = p_dict['agg']
        else:
            agg = self.agg
        if 'on' in p_dict.keys():
            on = p_dict['on']
        else:
            on = key
        return key, var, agg, on

    def _aggregate(self, dataframe):
        self.features = []
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[
                var].agg(agg).reset_index()
            features.columns = key + new_features
            self.features.append(features)
        return self

    def _merge(self, dataframe, merge=True):
        for param_dict, features in zip(self.param_dict, self.features):
            key, var, agg, on = self._get_params(param_dict)
            if merge:
                dataframe = dataframe.merge(features, how='left', on=on)
            else:
                new_features = self._get_feature_names(key, var, agg)
                dataframe = pd.concat([dataframe, features[new_features]], axis=1)
        return dataframe

    def transform(self, dataframe):
        self._aggregate(dataframe)
        return self._merge(dataframe, merge=True)

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join([a, v, 'groupby'] + key) for v in var for a in _agg]

    def get_feature_names(self):
        self.feature_names = []
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            self.feature_names += self._get_feature_names(key, var, agg)
        return self.feature_names

    def get_numerical_features(self):
        return self.get_feature_names()


class DiffGroupbyTransformer(GroupbyTransformer):
    def _aggregate(self):
        raise NotImplementedError

    def _merge(self):
        raise NotImplementedError

    def transform(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            for a in agg:
                for v in var:
                    new_feature = '_'.join(['diff', a, v, 'groupby'] + key)
                    base_feature = '_'.join([a, v, 'groupby'] + key)
                    dataframe[new_feature] = dataframe[base_feature] - dataframe[v]
        return dataframe

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join(['diff', a, v, 'groupby'] + key) for v in var for a in _agg]


class RatioGroupbyTransformer(GroupbyTransformer):
    def _aggregate(self):
        raise NotImplementedError

    def _merge(self):
        raise NotImplementedError

    def transform(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            for a in agg:
                for v in var:
                    new_feature = '_'.join(['ratio', a, v, 'groupby'] + key)
                    base_feature = '_'.join([a, v, 'groupby'] + key)
                    dataframe[new_feature] = dataframe[v] / dataframe[base_feature]
        return dataframe

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join(['ratio', a, v, 'groupby'] + key) for v in var for a in _agg]


class CategoryVectorizer():
    def __init__(self, categorical_columns, n_components,
                 vectorizer=CountVectorizer(),
                 transformer=LatentDirichletAllocation(),
                 name='CountLDA'):
        self.categorical_columns = categorical_columns
        self.n_components = n_components
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.name = name + str(self.n_components)

    def transform(self, dataframe):
        features = []
        for (col1, col2) in self.get_column_pairs():
            try:
                sentence = self.create_word_list(dataframe, col1, col2)
                sentence = self.vectorizer.fit_transform(sentence)
                feature = self.transformer.fit_transform(sentence)
                feature = self.get_feature(dataframe, col1, col2, feature, name=self.name)
                features.append(feature)
            except:
                pass
        features = pd.concat(features, axis=1)
        return features

    def create_word_list(self, dataframe, col1, col2):
        col1_size = int(dataframe[col1].values.max() + 1)
        col2_list = [[] for _ in range(col1_size)]
        for val1, val2 in zip(dataframe[col1].values, dataframe[col2].values):
            col2_list[int(val1)].append(col2 + str(val2))
        return [' '.join(map(str, ls)) for ls in col2_list]

    def get_feature(self, dataframe, col1, col2, latent_vector, name=''):
        features = np.zeros(shape=(len(dataframe), self.n_components), dtype=np.float32)
        self.columns = ['_'.join([name, col1, col2, str(i)]) for i in range(self.n_components)]
        for i, val1 in enumerate(dataframe[col1]):
            features[i, :self.n_components] = latent_vector[val1]

        return pd.DataFrame(data=features, columns=self.columns)

    def get_column_pairs(self):
        return [(col1, col2) for col1, col2 in itertools.product(self.categorical_columns, repeat=2) if col1 != col2]

    def get_numerical_features(self):
        return self.columns


def aggregation(train: pd.DataFrame,
                test: pd.DataFrame,
                groupby_dict: dict,
                nunique_dict: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregation

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        groupby_dict (dict): settings for groupby
        nunique_dict (dict): settings for nunique

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    groupby = GroupbyTransformer(param_dict=nunique_dict)
    train = groupby.transform(train)

    groupby = GroupbyTransformer(param_dict=groupby_dict)
    train = groupby.transform(train)

    diff = DiffGroupbyTransformer(param_dict=groupby_dict)
    train = diff.transform(train)

    ratio = RatioGroupbyTransformer(param_dict=groupby_dict)
    train = ratio.transform(train)

    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def matrix_factorization(train: pd.DataFrame,
                         test: pd.DataFrame,
                         encode_col: List[str],
                         n_components_lda: int = 5,
                         n_components_svd: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Matrix factorization

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns
        n_components_lda (int, optional): the output dimensions for lda. Defaults to 5.
        n_components_svd (int, optional): the output dimensions for svd. Defaults to 3.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    cf = CategoryVectorizer(encode_col,
                            n_components_lda,
                            vectorizer=CountVectorizer(),
                            transformer=LatentDirichletAllocation(n_components=n_components_lda,
                                                                  n_jobs=-1, learning_method='online', random_state=777),
                            name='CountLDA')
    features_lda = cf.transform(train).astype(np.float32)

    cf = CategoryVectorizer(encode_col,
                            n_components_svd,
                            vectorizer=CountVectorizer(),
                            transformer=TruncatedSVD(n_components=n_components_svd, random_state=777),
                            name='CountSVD')
    features_svd = cf.transform(train).astype(np.float32)

    train = pd.concat([train, features_svd, features_lda], axis=1)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def target_encoding(train: pd.DataFrame,
                    test: pd.DataFrame,
                    encode_col: List[str],
                    target_col: str,
                    cv) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Target encoding

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns
        target_col (str): target column
        cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    warnings.simplefilter('ignore')

    te = TargetEncoder(cv=cv)

    train_fe = te.fit_transform(train[encode_col], train[target_col])
    train_fe.columns = ['te_' + c for c in train_fe.columns]
    train = pd.concat([train, train_fe], axis=1)

    test_fe = te.transform(test[encode_col])
    test_fe.columns = ['te_' + c for c in test_fe.columns]
    test = pd.concat([test, test_fe], axis=1)

    return train, test


def frequency_encoding(train: pd.DataFrame,
                       test: pd.DataFrame,
                       encode_col: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Frequency encoding

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in encode_col:
        grouped = train.groupby(f).size().reset_index(name=f'fe_{f}')
        train = train.merge(grouped, how='left', on=f)
        train[f'fe_{f}'] = train[f'fe_{f}'] / train[f'fe_{f}'].count()

    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def count_encoding(train: pd.DataFrame,
                   test: pd.DataFrame,
                   encode_col: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Count encoding

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in encode_col:
        count_map = train[f].value_counts().to_dict()
        train[f'ce_{f}'] = train[f].map(count_map)

    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def count_encoding_interact(train: pd.DataFrame,
                            test: pd.DataFrame,
                            encode_col: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Count encoding for interaction

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for col1, col2 in list(itertools.combinations(encode_col, 2)):
        col = col1 + '_' + col2
        _tmp = train[col1].astype(str) + "_" + train[col2].astype(str)
        count_map = _tmp.value_counts().to_dict()
        train[f'cei_{col}'] = _tmp.map(count_map)

    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def numeric_interact(train: pd.DataFrame,
                     test: pd.DataFrame,
                     encode_col: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Numerical interaction

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for col1, col2 in list(itertools.combinations(encode_col, 2)):
        train[f'{col1}_plus_{col2}'] = train[col1] + train[col2]
        train[f'{col1}_mul_{col2}'] = train[col1] * train[col2]
        try:
            train[f'{col1}_div_{col2}'] = train[col1] / train[col2]
        except:
            print(f'{col1}_div_{col2}')

    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def count_null(train: pd.DataFrame,
               test: pd.DataFrame,
               encode_col: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Count NaN

    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        encode_col (List[str]): encoded columns

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    train['count_null'] = train.isnull().sum(axis=1)
    for f in encode_col:
        if sum(train[f].isnull().astype(int)) > 0:
            train[f'cn_{f}'] = train[f].isnull().astype(int)

    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test
