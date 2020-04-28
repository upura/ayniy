from os.path import join
import pandas as pd
import numpy as np
from ayniy.utils import timer
from ayniy.preprocessing.tabular import count_null
from ayniy.preprocessing.tabular import label_encoding
from ayniy.preprocessing.tabular import frequency_encoding
from ayniy.preprocessing.tabular import count_encoding
from ayniy.preprocessing.tabular import count_encoding_interact
from ayniy.preprocessing.tabular import matrix_factorization
from ayniy.preprocessing.tabular import target_encoding
from ayniy.preprocessing.tabular import aggregation
from ayniy.preprocessing.tabular import numeric_interact
from ayniy.preprocessing.tabular import delete_cols
from ayniy.preprocessing.tabular import detect_delete_cols
from ayniy.preprocessing.tabular import save_as_pickle
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

    def create(self) -> None:
        with timer('null counting'):
            encode_col = list(self.train.columns)
            encode_col.remove(self.cols_definition['target_col'])
            train, test = count_null(self.train, self.test, {'encode_col': encode_col})

        with timer('label encoding'):
            categorical_col = self.cols_definition['categorical_col']
            self.train, self.test = label_encoding(self.train, self.test, {'encode_col': categorical_col})

        with timer('frequency encoding'):
            self.train, self.test = frequency_encoding(self.train, self.test, {'encode_col': categorical_col})

        with timer('count encoding'):
            self.train, self.test = count_encoding(self.train, self.test, {'encode_col': categorical_col})

        with timer('count encoding interact'):
            self.train, self.test = count_encoding_interact(self.train, self.test, {'encode_col': categorical_col})

        if 'matrix_factorization' in self.preprocessing.keys():
            with timer('frequency encoding'):
                self.train, self.test = matrix_factorization(
                    self.train, self.test,
                    {'encode_col': self.preprocessing['matrix_factorization']},
                    {'n_components_lda': 5, 'n_components_svd': 3})

        if 'target_encoding' in self.preprocessing.keys():
            with timer('target encoding'):
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

        with timer('numeric interact'):
            self.train, self.test = numeric_interact(self.train, self.test, {'encode_col': self.cols_definition['numerical_col']})

        with timer('replace inf'):
            self.train = self.train.replace(np.inf, 9999999999).replace(-np.inf, -9999999999)
            self.test = self.test.replace(np.inf, 9999999999).replace(-np.inf, -9999999999)

        with timer('delete cols'):
            unique_cols, duplicated_cols, high_corr_cols = detect_delete_cols(
                self.train, self.test, {'escape_col': self.cols_definition['categorical_col']}, {'threshold': 0.995})
            print('unique_cols: ', unique_cols)
            print('duplicated_cols: ', duplicated_cols)
            print('high_corr_cols: ', high_corr_cols)
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
