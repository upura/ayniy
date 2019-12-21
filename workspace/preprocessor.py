import yaml
import sys
from os.path import join
import pandas as pd
from ayniy.utils import timer
from ayniy.preprocessing.tabular import use_cols, delete_cols, label_encoding
from ayniy.preprocessing.tabular import aggregation, matrix_factorization
from ayniy.preprocessing.tabular import save_as_pickle


if __name__ == '__main__':

    with timer('loading'):
        args = sys.argv
        configs = yaml.load(open(join('configs', f'{args[1]}.yml'), 'r+'))

        input_dir = join('../input', configs['data']['input'])
        train = pd.read_csv(join(input_dir, 'train.csv'))
        test = pd.read_csv(join(input_dir, 'test.csv'))
        print(f'train.shape: {train.shape}')

        id_col = configs['cols_definition']['id_col']
        target_col = configs['cols_definition']['target_col']

    with timer('preprocessing'):
        if 'use_columns' in configs['preprocessing']:
            train, test = use_cols(train, test,
                                   configs['preprocessing']['use_columns'],
                                   configs['cols_definition']['target_col'])
        if 'delete_columns' in configs['preprocessing']:
            train, test = delete_cols(train, test, configs['preprocessing']['delete_columns'])
        if 'label_encoding' in configs['preprocessing']:
            train, test = label_encoding(train, test, configs['preprocessing']['label_encoding'])
        if 'aggregation' in configs['preprocessing']:
            train, test = aggregation(train, test,
                                      configs['preprocessing']['aggregation']['groupby_dict'],
                                      configs['preprocessing']['aggregation']['nunique_dict'])
        if 'matrix_factorization' in configs['preprocessing']:
            train, test = matrix_factorization(train, test,
                                               configs['preprocessing']['matrix_factorization'])

    with timer('saving'):
        save_as_pickle(train, test, configs['cols_definition']['target_col'])
        print(f'train.shape: {train.shape}')
