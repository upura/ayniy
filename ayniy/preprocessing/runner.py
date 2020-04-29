from os.path import join
import pandas as pd
import numpy as np
from ayniy.utils import timer
from ayniy.preprocessing.tabular import (count_null,
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
                                         save_as_pickle)
from ayniy.preprocessing.text import get_tfidf, get_count, get_swem, get_scdv
from ayniy.utils import Data


class Tabular:

    def __init__(self, configs: dict, cv=None):
        self.run_name = configs['run_name']
        self.train = pd.read_csv(configs['data']['train'])
        self.test = pd.read_csv(configs['data']['test'])
        self.output_dir = configs['data']['output_dir']
        self.cols_definition = configs['cols_definition']
        self.preprocessing = configs['preprocessing']
        self.cv = cv
        self.logger = {}

    def create(self) -> None:

        if 'count_null' in self.preprocessing.keys():
            with timer('count_null'):
                encode_col = list(self.train.columns)
                encode_col.remove(self.cols_definition['target_col'])
                train, test = count_null(self.train, self.test, {'encode_col': encode_col})

        if 'label_encoding' in self.preprocessing.keys():
            with timer('label_encoding'):
                self.train, self.test = label_encoding(self.train, self.test, {'encode_col': self.cols_definition['categorical_col']})

        if 'frequency_encoding' in self.preprocessing.keys():
            with timer('frequency_encoding'):
                self.train, self.test = frequency_encoding(self.train, self.test, {'encode_col': self.cols_definition['categorical_col']})

        if 'count_encoding' in self.preprocessing.keys():
            with timer('count_encoding'):
                self.train, self.test = count_encoding(self.train, self.test, {'encode_col': self.cols_definition['categorical_col']})

        if 'count_encoding_interact' in self.preprocessing.keys():
            with timer('count_encoding_interact'):
                self.train, self.test = count_encoding_interact(self.train, self.test, {'encode_col': self.cols_definition['categorical_col']})

        if 'matrix_factorization' in self.preprocessing.keys():
            with timer('matrix_factorization'):
                self.train, self.test = matrix_factorization(
                    self.train, self.test,
                    {'encode_col': self.preprocessing['matrix_factorization']},
                    {'n_components_lda': 5, 'n_components_svd': 3})

        if 'target_encoding' in self.preprocessing.keys():
            with timer('target_encoding'):
                self.train, self.test = target_encoding(
                    self.train, self.test,
                    {'encode_col': self.preprocessing['target_encoding'],
                     'target_col': self.cols_definition['target_col']},
                    {'cv': self.cv})

        if 'aggregation' in self.preprocessing.keys():
            with timer('aggregation'):
                self.train, self.test = aggregation(
                    self.train, self.test,
                    {'groupby_dict': self.preprocessing['aggregation']['groupby_dict'],
                     'nunique_dict': self.preprocessing['aggregation']['nunique_dict']})

        if 'numeric_interact' in self.preprocessing.keys():
            with timer('numeric_interact'):
                self.train, self.test = numeric_interact(self.train, self.test, {'encode_col': self.cols_definition['numerical_col']})

        if 'get_tfidf' in self.preprocessing.keys():
            with timer('get_tfidf'):
                for tc in self.cols_definition['text_col']:
                    self.train, self.test = get_tfidf(self.train, self.test,
                                                      {'text_col': tc, 'target_col': self.cols_definition['target_col']},
                                                      self.preprocessing['get_tfidf'])

        if 'get_count' in self.preprocessing.keys():
            with timer('get_count'):
                for tc in self.cols_definition['text_col']:
                    self.train, self.test = get_count(self.train, self.test,
                                                      {'text_col': tc, 'target_col': self.cols_definition['target_col']},
                                                      self.preprocessing['get_count'])

        with timer('replace inf'):
            self.train = self.train.replace(np.inf, 9999999999).replace(-np.inf, -9999999999)
            self.test = self.test.replace(np.inf, 9999999999).replace(-np.inf, -9999999999)

        with timer('delete cols'):
            unique_cols, duplicated_cols, high_corr_cols = detect_delete_cols(
                self.train, self.test, {'escape_col': self.cols_definition['categorical_col']}, {'threshold': 0.995})
            self.logger['unique_cols'] = unique_cols
            self.logger['duplicated_cols'] = duplicated_cols
            self.logger['high_corr_cols'] = high_corr_cols
            self.train, self.test = delete_cols(
                self.train, self.test,
                {'encode_col': unique_cols + duplicated_cols + high_corr_cols + self.cols_definition['delete_col']})

        with timer('save'):
            save_as_pickle(self.train, self.test,
                           {'target_col': self.cols_definition['target_col']},
                           {'exp_id': self.run_name, 'output_dir': self.output_dir})

    def load(self):
        X_train = Data.load(join(self.output_dir, f"X_train_{self.run_name}.pkl"))
        y_train = Data.load(join(self.output_dir, f"y_train_{self.run_name}.pkl"))
        X_test = Data.load(join(self.output_dir, f"X_test_{self.run_name}.pkl"))
        return X_train, X_test, y_train
