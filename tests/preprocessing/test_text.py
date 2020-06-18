import pytest

from ayniy.preprocessing.text import (get_tfidf,
                                      get_count,
                                      get_bert,
                                      get_swem_mean)


def test_en_get_tfidf(load_en_text):
    train, test = load_en_text
    train_new, _ = get_tfidf(train, test,
                             {'text_col': 'text'},
                             {'n_components': 5, 'lang': 'en'})
    assert len((set(train_new.columns) - set(train.columns))) > 0


def test_en_get_count(load_en_text):
    train, test = load_en_text
    train_new, _ = get_count(train, test,
                             {'text_col': 'text'},
                             {'n_components': 5, 'lang': 'en'})
    assert len((set(train_new.columns) - set(train.columns))) > 0


@pytest.mark.skip
def test_en_get_bert(load_en_text):
    train, test = load_en_text
    train_new, _ = get_bert(train, test,
                            {'text_col': 'text'},
                            {'n_components': 5, 'lang': 'en'})
    assert len((set(train_new.columns) - set(train.columns))) > 0


@pytest.mark.skip
def test_en_get_swem_mean(load_en_text):
    train, test = load_en_text
    train_new, _ = get_swem_mean(train, test,
                                 {'text_col': 'text'},
                                 {'n_components': 5, 'lang': 'en'})
    assert len((set(train_new.columns) - set(train.columns))) > 0


def test_ja_get_tfidf(load_ja_text):
    train, test = load_ja_text
    train_new, _ = get_tfidf(train, test,
                             {'text_col': 'text'},
                             {'n_components': 5, 'lang': 'ja'})
    assert len((set(train_new.columns) - set(train.columns))) > 0


def test_ja_get_count(load_ja_text):
    train, test = load_ja_text
    train_new, _ = get_count(train, test,
                             {'text_col': 'text'},
                             {'n_components': 5, 'lang': 'ja'})
    assert len((set(train_new.columns) - set(train.columns))) > 0


@pytest.mark.skip
def test_ja_get_bert(load_ja_text):
    train, test = load_ja_text
    train_new, _ = get_bert(train, test,
                            {'text_col': 'text'},
                            {'n_components': 5, 'lang': 'ja'})
    assert len((set(train_new.columns) - set(train.columns))) > 0


@pytest.mark.skip
def test_ja_get_swem_mean(load_ja_text):
    train, test = load_ja_text
    train_new, _ = get_swem_mean(train, test,
                                 {'text_col': 'text'},
                                 {'n_components': 5, 'lang': 'ja'})
    assert len((set(train_new.columns) - set(train.columns))) > 0
