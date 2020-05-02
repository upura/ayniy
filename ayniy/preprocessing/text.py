import numpy as np
import pandas as pd
import re
from tqdm import tqdm_notebook as tqdm
import neologdn
from sklearn.pipeline import make_pipeline, make_union
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import scipy as sp
import nltk
from sklearn.feature_extraction.text import _document_frequency
from sklearn.utils.validation import check_is_fitted
from transformers import BertTokenizer, BertJapaneseTokenizer, BertModel
import torch
from torch.utils.data import DataLoader


def analyzer_bow_en(text):
    sb = nltk.stem.snowball.SnowballStemmer('english')
    stop_words = ['i', 'a', 'an', 'the', 'to', 'and', 'or', 'if', 'is', 'are', 'am', 'it', 'this', 'that', 'of', 'from', 'in', 'on']
    text = text.lower()     # 小文字化
    text = text.replace('\n', '')   # 改行削除
    text = text.replace('\t', '')   # タブ削除
    puncts = r',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√。【】'
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    text = text.split(' ')  # スペースで区切る
    text = [sb.stem(t) for t in text]

    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None):   # 数字が含まれるものは分割
            for w in re.findall(r'(\d+|\D+)', word):
                words.append(w)
            continue
        if word in stop_words:  # ストップワードに含まれるものは除外
            continue
        if len(word) < 2:   # 1文字、0文字（空文字）は除外
            continue
        words.append(word)

    return ' '.join(words)


class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b  : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        """
        if not sp.sparse.issparse(X):
            X = sp.sparse.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.sparse.spdiags(idf, diags=0, m=n_features, n=n_features)

        doc_len = X.sum(axis=1)
        self._average_document_len = np.average(doc_len)

        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.sparse.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.sparse.csr_matrix(X, dtype=np.float, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        doc_len = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]

        # In each row, repeat `doc_len` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(doc_len), sz)

        # Compute BM25 score only for non-zero elements
        nom = self.k1 + 1
        denom = X.data + self.k1 * (1 - self.b + self.b * rep / self._average_document_len)
        data = X.data * nom / denom

        X = sp.sparse.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = X * self._idf_diag

        return X


def text_normalize(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict):
    """
    col_definition: text_col
    """
    train[col_definition['text_col']] = train[col_definition['text_col']].apply(neologdn.normalize)
    test[col_definition['text_col']] = test[col_definition['text_col']].apply(neologdn.normalize)
    return train, test


def get_tfidf(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict, option: dict):
    """
    col_definition: text_col
    option: n_components, lang={'ja', 'en'}
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)
    vectorizer = make_pipeline(
        TfidfVectorizer(),
        make_union(
            TruncatedSVD(n_components=option['n_components'], random_state=7),
            NMF(n_components=option['n_components'], random_state=7),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=option['n_components'], random_state=7)
            ),
            n_jobs=1,
        ),
    )

    if option['lang'] == 'en':
        X = [analyzer_bow_en(text) for text in train[col_definition['text_col']].fillna('')]
    else:
        raise ValueError
    X = vectorizer.fit_transform(X).astype(np.float32)
    X = pd.DataFrame(X, columns=[
        'tfidf_svd_{}'.format(i) for i in range(option['n_components'])] + [
        'tfidf_nmf_{}'.format(i) for i in range(option['n_components'])] + [
        'tfidf_bm25_{}'.format(i) for i in range(option['n_components'])])
    train = pd.concat([train, X], axis=1)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def get_count(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict, option: dict):
    """
    col_definition: text_col
    option: n_components, lang={'ja', 'en'}
    """

    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)
    vectorizer = make_pipeline(
        CountVectorizer(min_df=2, max_features=20000,
                        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                        ngram_range=(1, 3), stop_words='english'),
        make_union(
            TruncatedSVD(n_components=option['n_components'], random_state=7),
            NMF(n_components=option['n_components'], random_state=7),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=option['n_components'], random_state=7)
            ),
            n_jobs=1,
        ),
    )

    if option['lang'] == 'en':
        X = [analyzer_bow_en(text) for text in train[col_definition['text_col']].fillna('')]
    else:
        raise ValueError
    X = vectorizer.fit_transform(X).astype(np.float32)
    X = pd.DataFrame(X, columns=[
        'count_svd_{}'.format(i) for i in range(option['n_components'])] + [
        'count_nmf_{}'.format(i) for i in range(option['n_components'])] + [
        'count_bm25_{}'.format(i) for i in range(option['n_components'])])
    train = pd.concat([train, X], axis=1)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def get_bert(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict, option: dict):
    """
    col_definition: text_col
    option: n_components, lang={'ja', 'en'}, batch_size
    """

    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)
    vectorizer = make_pipeline(
        make_union(
            TruncatedSVD(n_components=option['n_components'], random_state=7),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=option['n_components'], random_state=7)
            ),
            n_jobs=1,
        ),
    )

    if option['lang'] == 'en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif option['lang'] == 'ja':
        tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese')
        model = BertModel.from_pretrained('bert-base-japanese')
    else:
        raise ValueError

    X = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    encoded_data = tokenizer.batch_encode_plus(
        list(train[col_definition['text_col']].fillna('').values),
        pad_to_max_length=True,
        add_special_tokens=True)
    input_ids = torch.tensor(encoded_data['input_ids'])
    loader = DataLoader(input_ids, batch_size=option['batch_size'], shuffle=False)
    for x in tqdm(loader, leave=False):
        x = x.to(device)
        outputs = model(x)
        X.append(outputs[0][:, 0, :])
    X = torch.cat(X)
    X = X.detach().cpu().numpy()
    X = vectorizer.fit_transform(X).astype(np.float32)

    X = pd.DataFrame(X, columns=[
        'bert_svd_{}'.format(i) for i in range(option['n_components'])] + [
        'bert_bm25_{}'.format(i) for i in range(option['n_components'])])
    train = pd.concat([train, X], axis=1)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test
