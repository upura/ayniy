import seaborn as sns
import numpy as np

from ayniy.preprocessing.tabular import label_encoding


def load_titanic():
    train = sns.load_dataset('titanic')
    test = train
    return train, test


def test_label_encoding():
    train, test = load_titanic()
    categorical_col = ['embarked', 'sex']
    train, test = label_encoding(train, test, {'encode_col': categorical_col})
    assert type(train[categorical_col[0]][0]) == np.int64
