import seaborn as sns
import pandas as pd
import pytest

_TEST_SENTENCE_EN = [
    'This is a pen.',
    'A quick brown fox',
    'Redistribution and use in source and binary forms, with or without modification.',
    'BERT is the state of the art NLP model.',
    'This is a pen.',
    'THIS IS A PEN.',
]

_TEST_SENTENCE_JP = [
    '金メダルが5枚欲しい。',
    '私は昨日から風邪をひいています。',
    'これはペンです。',
    'BERTは最新の自然言語処理モデルです。',
    '金メダルが5枚欲しい。',
    '金メダルが 5枚 欲しい。',
]


@pytest.fixture
def load_titanic():
    train = sns.load_dataset('titanic')
    test = train.copy()
    return train, test


@pytest.fixture
def load_en_text():
    train = pd.DataFrame({'text': _TEST_SENTENCE_EN})
    test = train.copy()
    return train, test


@pytest.fixture
def load_ja_text():
    train = pd.DataFrame({'text': _TEST_SENTENCE_JP})
    test = train.copy()
    return train, test
