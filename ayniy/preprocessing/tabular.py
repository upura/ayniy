import pandas as pd
import numpy as np
from sklearn import preprocessing
from os.path import join
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.decomposition import PCA
import itertools
from kaggler.preprocessing import TargetEncoder

from ayniy.utils import Data


def use_cols(train, test, include_cols, target_col):
    train = train[include_cols + [target_col]]
    test = test[include_cols]
    return train, test


def delete_cols(train, test, exclude_cols):
    train.drop(exclude_cols, inplace=True, axis=1)
    test.drop(exclude_cols, inplace=True, axis=1)
    return train, test


def label_encoding(train, test, encode_cols):
    for f in encode_cols:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
    return train, test


def standerize(train, test, numerical_cols=None):
    scaler = preprocessing.StandardScaler()

    if numerical_cols is None:
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    else:
        train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
        test[numerical_cols] = scaler.transform(test[numerical_cols])
    return pd.DataFrame(train), pd.DataFrame(test)


def fillna(train, test, encode_cols, how='median'):
    for f in encode_cols:
        if how == 'median':
            test[f].fillna(train[f].median(), inplace=True)
            test[f].fillna(train[f].median(), inplace=True)
        elif how == 'mean':
            test[f].fillna(train[f].mean(), inplace=True)
            test[f].fillna(train[f].mean(), inplace=True)
    return pd.DataFrame(train), pd.DataFrame(test)


def datatime_parser(train, test, encode_col):
    train['year'] = pd.to_datetime(train[encode_col]).dt.year
    train['month'] = pd.to_datetime(train[encode_col]).dt.month
    train['day'] = pd.to_datetime(train[encode_col]).dt.day
    train['dow'] = pd.to_datetime(train[encode_col]).dt.dayofweek
    train['hour'] = pd.to_datetime(train[encode_col]).dt.hour
    train['minute'] = pd.to_datetime(train[encode_col]).dt.minute
    test['year'] = pd.to_datetime(test[encode_col]).dt.year
    test['month'] = pd.to_datetime(test[encode_col]).dt.month
    test['day'] = pd.to_datetime(test[encode_col]).dt.day
    test['dow'] = pd.to_datetime(test[encode_col]).dt.dayofweek
    test['hour'] = pd.to_datetime(test[encode_col]).dt.hour
    test['minute'] = pd.to_datetime(test[encode_col]).dt.minute
    return train, test


def circle_encoding(train, test, encode_cols):
    for f in encode_cols:
        train[f + '_cos'] = np.cos(2 * np.pi * train[f] / train[f].max())
        train[f + '_sin'] = np.sin(2 * np.pi * train[f] / train[f].max())
        test[f + '_cos'] = np.cos(2 * np.pi * test[f] / train[f].max())
        test[f + '_sin'] = np.sin(2 * np.pi * test[f] / train[f].max())
    return train, test


def save_as_pickle(train, test, target_col, out_dir='outputs'):

    X_train = train.drop(target_col, axis=1)
    y_train = train[target_col]
    if target_col in test.columns:
        X_test = test.drop(target_col, axis=1)
    else:
        X_test = test

    Data.dump(X_train, join(out_dir, 'X_train.pkl'))
    Data.dump(y_train, join(out_dir, 'y_train.pkl'))
    Data.dump(X_test, join(out_dir, 'X_test.pkl'))


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


def aggregation(train, test, groupby_dict, nunique_dict,
                stats=['mean', 'sum', 'median', 'min', 'max', 'var', 'std']):

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


def matrix_factorization(train, test, categorical_features,
                         n_components_lda=5, n_components_svd=3):
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    cf = CategoryVectorizer(categorical_features, n_components_lda,
                            vectorizer=CountVectorizer(),
                            transformer=LatentDirichletAllocation(n_components=n_components_lda, n_jobs=-1, learning_method='online', random_state=777),
                            name='CountLDA')
    features_lda = cf.transform(train).astype(np.float32)

    cf = CategoryVectorizer(categorical_features, n_components_svd,
                            vectorizer=CountVectorizer(),
                            transformer=TruncatedSVD(n_components=n_components_svd, random_state=777),
                            name='CountSVD')
    features_svd = cf.transform(train).astype(np.float32)

    train = pd.concat([train, features_svd, features_lda], axis=1)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def max_devided_min(x):
    return max(x) / (min(x) + 0.000001)


def max_minus_min(x):
    return max(x) - min(x)


def target_encoding_train(X_train, key_col, folds_path, stats):
    folds = Data.load(folds_path)
    dfs = []

    for train_index, test_index in folds:
        X_used, X_unused = X_train.loc[train_index, :], X_train.loc[test_index, :]

        agg = X_used.groupby(key_col).agg(stats)
        names_1 = agg.columns.get_level_values(0)
        names_2 = agg.columns.get_level_values(1)
        agg.columns = ['agg_' + key_col + '_' + el1 + '_' + el2 for (el1, el2) in zip(names_1, names_2)]
        tmp = pd.merge(X_unused, agg, on=key_col, how='left')
        tmp.index = test_index
        dfs.append(tmp)

    return pd.concat(dfs).sort_index()


def target_encoding_test(X_train, X_test, key_col, stats):
    agg = X_train.groupby(key_col).agg(stats)
    names_1 = agg.columns.get_level_values(0)
    names_2 = agg.columns.get_level_values(1)
    agg.columns = ['agg_' + key_col + '_' + el1 + '_' + el2 for (el1, el2) in zip(names_1, names_2)]
    return pd.merge(X_test, agg, on=key_col, how='left')


def target_encoding(train, test, categorical_features, target_col, folds_path,
                    stats={'sum', 'min', 'mean', 'median', 'max', 'var', 'quantile', max_devided_min, max_minus_min}):
    train_fe = []
    test_fe = []

    for c in categorical_features:
        tmp_train = target_encoding_train(train[categorical_features + [target_col]],
                                          c, folds_path, stats).drop(categorical_features + [target_col], axis=1)
        train_fe.append(tmp_train)
        tmp_test = target_encoding_test(train[categorical_features + [target_col]], test[categorical_features],
                                        c, stats).drop(categorical_features, axis=1)
        test_fe.append(tmp_test)

    train = pd.concat([train] + train_fe, axis=1)
    test = pd.concat([test] + test_fe, axis=1)
    return train, test


def pca_df(df, hidden_dims):
    pca_clf = PCA(n_components=hidden_dims)
    return pca_clf.fit_transform(df)


def target_encoding_smoothing(train, test, categorical_features, target_col, cv):
    te = TargetEncoder(cv=cv)
    train.loc[:, categorical_features] = te.fit_transform(train[categorical_features], train[target_col])
    test.loc[:, categorical_features] = te.transform(test[categorical_features])
    return train, test
