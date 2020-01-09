from __future__ import unicode_literals
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.pipeline import make_pipeline, make_union
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import unicodedata
import MeCab
import os
import requests
from tqdm import tqdm
from collections import defaultdict
import zipfile
import gensim
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, TransformerMixin
import scipy as sp
from sklearn.feature_extraction.text import _document_frequency
from sklearn.utils.validation import check_is_fitted


"""
ストップワード, 品詞の条件に基づいて
単語をコーパスに載せるかを判断するstoperの定義
"""
WORDS_FOR_CONTENTS = [
    # 形容詞のストップワード
    # 固有の意味が薄く多用されるワード
    "ない",
    "高い",
    "多い",
    "少ない",
    "強い",
    "大きい",
    "小さい",
    "良い",

    # 動詞のストップワード
    "ある",
    "いる",
    "なる",
    "行く",
    "いる",
    "とる",
    "見る",
    "みる",
    "言う",
    "いう",
    "得る",
    "過ぎる",
    "すぎる",
    "する",
    "やる",
]

HINSHI_FOR_CONTENTS = {
    "品詞": ["連体詞", "接続詞", "助詞", "助動詞", "連語", "副詞", "接頭語"],
    "品詞細分類1": ["形容動詞語幹", "副詞可能", "代名詞", "ナイ形容詞互換", "特殊", "数", "接尾", "非自立"],
}
HANKAKU_PATTARN = r"[!-/:-@[-`{-~]"

sb = nltk.stem.snowball.SnowballStemmer('english')


def analyzer_bow(text):
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

    return " ".join(words)


def analyzer_embed(text):
    text = text.lower()     # 小文字化
    text = text.replace('\n', '')   # 改行削除
    text = text.replace('\t', '')   # タブ削除
    puncts = r',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√。【】'
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    text = text.split(' ')  # スペースで区切る

    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None):   # 数字が含まれるものは分割
            for w in re.findall(r'(\d+|\D+)', word):
                words.append(w)
            continue
        if len(word) < 1:   # 0文字（空文字）は除外
            continue
        words.append(word)
    return " ".join(words)


def get_tfidf_svd_nmf_bm25(train, test, text_col):
    n_components = 5
    seed = 777
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    vectorizer = make_pipeline(
        TfidfVectorizer(),
        make_union(
            TruncatedSVD(n_components=n_components, random_state=seed),
            NMF(n_components=n_components, random_state=seed),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=n_components, random_state=seed)
            ),
            n_jobs=1,
        ),
    )

    train[text_col + '_bow'] = [analyzer_bow(text) for text in train[text_col]]
    X = vectorizer.fit_transform(train[text_col + '_bow']).astype(np.float32)
    X = pd.DataFrame(X, columns=[
        'tfidf_svd_{}'.format(i) for i in range(n_components)] + [
        'tfidf_nmf_{}'.format(i) for i in range(n_components)] + [
        'tfidf_bm25_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)

    train.drop(text_col + '_bow', axis=1, inplace=True)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def get_count_svd_nmf_bm25(train, test, text_col):
    n_components = 5
    seed = 777
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    vectorizer = make_pipeline(
        CountVectorizer(min_df=2, max_features=20000,
                        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                        ngram_range=(1, 3), stop_words='english'),
        make_union(
            TruncatedSVD(n_components=n_components, random_state=seed),
            NMF(n_components=n_components, random_state=seed),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=n_components, random_state=seed)
            ),
            n_jobs=1,
        ),
    )

    train[text_col + '_bow'] = [analyzer_bow(text) for text in train[text_col]]
    X = vectorizer.fit_transform(train[text_col + '_bow']).astype(np.float32)
    X = pd.DataFrame(X, columns=[
        'count_svd_{}'.format(i) for i in range(n_components)] + [
        'count_nmf_{}'.format(i) for i in range(n_components)] + [
        'count_bm25_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)

    train.drop(text_col + '_bow', axis=1, inplace=True)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


class OchasenLine(object):
    def __init__(self, line):
        x = line.split('\t')
        try:
            # 単語そのもの: かかわり
            self.word = x[0]
            # 単語の読み方: カカワリ
            self.yomi = x[1]
            # 原型 かかわり -> かかわる
            self.norm_word = x[2]

            # 推定される品詞: 助詞-格助詞-一般
            hinshi = x[3].split('-')
            self.hinshi_class = hinshi[0]

            if len(hinshi) > 1:
                self.hinshi_detail = hinshi[1]
            else:
                self.hinshi_detail = None

            self.can_parse = True
        except Exception as e:
            self.can_parse = False
            print('not parse: {}'.format(line), e)
        self.line = line

    def __str__(self):
        if self.can_parse:
            return '{0.word}-{0.yomi}'.format(self)
        return None

    def __repr__(self):
        return self.__str__()


"""
文字の正規化
参考: https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja
"""


def unicode_normalize(cls, doc):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(codec):
        return unicodedata.normalize('NFKC', codec) if pt.match(codec) else codec

    doc = ''.join(norm(x) for x in re.split(pt, doc))
    doc = re.sub('－', '-', doc)
    return doc


def remove_extra_spaces(doc):
    """
    余分な空白を削除
    Args:
        doc (String)
    Return
        空白除去された文章 (String)
    """

    doc = re.sub('[ 　]+', ' ', doc)
    blocks = ''.join((
        '\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
        '\u3040-\u309F',  # HIRAGANA
        '\u30A0-\u30FF',  # KATAKANA
        '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
        '\uFF00-\uFFEF'  # HALFWIDTH AND FULLWIDTH FORMS
    ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, doc):
        pt = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while pt.search(doc):
            doc = pt.sub(r'\1\2', doc)
        return doc

    doc = remove_space_between(blocks, blocks, doc)
    doc = remove_space_between(blocks, basic_latin, doc)
    doc = remove_space_between(basic_latin, blocks, doc)
    return doc


def normalize_neologd(doc):
    """
    以下の文章の正規化を行います.
        * 空白の削除
        * 文字コードの変換(utf-8へ)
        * ハイフン,波線（チルダ)の統一
        * 全角記号の半角への変換   (？→?など)
    Args:
        doc(str):
            正規化を行いたい文章
    Return(str):
        正規化された文章
    """

    doc = doc.strip()
    doc = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', doc)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    doc = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', doc)  # normalize hyphens
    doc = re.sub('[﹣－ｰ—―─━ー]+', 'ー', doc)  # normalize choonpus
    doc = re.sub('[~∼∾〜〰～]', '', doc)  # remove tildes
    doc = doc.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･「」「」',
                  '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」『』'))

    doc = remove_extra_spaces(doc)
    doc = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', doc)  # keep ＝,・,「,」
    doc = re.sub('[’]', '\'', doc)
    doc = re.sub('[”]', '"', doc)
    doc = re.sub('[“]', '"', doc)
    return doc


class Stopper(object):
    """
    単語とクラスを受取り, 単語を残すかどうかを判定
    ストップワードや、特定の品詞を取り除くことを想定しています。
    """

    def __init__(self,
                 stop_hinshi=None,
                 stop_words=None,
                 remove_sign=True,
                 remove_oneword=True):
        """
        Args:
            stop_hinshi (dictionary | str)
                取り除く品詞の入ったディクショナリ。
                Noneのときはデフォルト値として空のリスト`[]`を用います.
            stop_words (List<string> | str)
                ストップワードのリスト.
                Noneのときはデフォルトとして空のリスト`[]`を用います.
            remove_sign (boolean)
                記号を除去するかどうかのboolean. デフォルト値はTrue.
            remove_oneword (boolean)
                一文字のwordを取り除くかどうかのboolean.
                機械翻訳がタスクの場合、係り受けを考慮しなくてはならないので, 一字であっても除去しないのが普通です。
                文章の内容の要約がタスクの場合、一字の言葉は内容を表していない場合が多くあるという観点から除去する場合があります。
                要するにタスク依存です。これは記号除去にも同じことがいえます。
        """

        if stop_hinshi is None:
            self.stop_hinshi = {}
        elif stop_hinshi == "contents":
            self.stop_hinshi = HINSHI_FOR_CONTENTS
        else:
            self.stop_hinshi = stop_hinshi

        if stop_words is None:
            self.stop_words = []
        elif stop_words == "contents":
            self.stop_words = WORDS_FOR_CONTENTS
        else:
            self.stop_words = stop_words

        self.remove_sign = remove_sign
        self.remove_oneword = remove_oneword

    def is_oneword(self, word):
        """
        単語が一文字であるかどうか判定
        Args:
            word (string)
        Returns:
            word is constructed with one word or not (Boolean)
        """

        if len(word) == 1:
            return True
        else:
            return False

    def __call__(self, word, word_class):
        """
        Args:
            word:
            word_class:
        Returns:
            boolean
        """

        if self.remove_oneword and self.is_oneword(word):
            return False

        if word in self.stop_words:
            return False

        for _, value in self.stop_hinshi.items():
            for w_class in word_class:
                if w_class in value:
                    return False
        return True


class DocumentParser(object):
    def __init__(self, stopper=None, as_normed=True):
        """
        Args:
            stopper(Stopper | None):
                stop word を拡張した Stopper instance.
                分かち書き結果に含めたくない単語などが有る場合, stopper クラスを渡す.
                特に指定がない場合すべての単語を分かち書き結果に含める.
            as_normed(bool):
                True のとき原型を分かち書きとして返す.
                文章の意味解析の場合活用はあまり考慮する必要が無いため指定すると吉
        Examples:
            In [2]: parser = DocumentParser()
            In [3]: s = '買えそう'
            In [4]: parser.call(s)
            Out[4]: ['買える', 'そう']
        """

        self.tagger = MeCab.Tagger('-Ochasen')

        if stopper is None:
            stopper = Stopper()
        self.stopper = stopper
        self.as_normed = as_normed

    def get_word(self, ocha):
        """
        Ochasen でわけられた OchasenLine から単語を取得する
        Args:
            ocha(OchasenLine):
        Returns(str):
        """
        if self.as_normed:
            return ocha.norm_word
        else:
            return ocha.word

    def is_valid_line(self, ocha):
        """
        Args:
            ocha(OchasenLine):
        Returns(bool):
        """
        if self.stopper is None:
            return ocha.can_parse

        return ocha.can_parse and self.stopper(ocha.norm_word, ocha.hinshi_class)

    def call(self, sentence):
        """
        文章の文字列を受け取り分かち書きされた list を返す
        Args:
            sentence(str):
        Returns:
            list[str]
        """
        s = normalize_neologd(sentence)
        # 文字列への事前処理を pytorch の compose っぽく書きたい
        s = s.lower()
        lines = self.tagger.parse(s).splitlines()[:-1]
        ocha_lines = [OchasenLine(l) for l in lines]

        return [self.get_word(ocha) for ocha in ocha_lines if self.is_valid_line(ocha)]


def create_parsed_document(docs):
    parser = DocumentParser(stopper=Stopper(stop_hinshi='contents'))
    parsed_docs = [parser.call(s) for s in docs]
    return parsed_docs


def create_idf_dataframe(documents):
    """
    Args:
        documents(list[str]):
    Returns(pd.DataFrame):
    """

    d = defaultdict(int)

    for doc in documents:
        vocab_i = set(doc)
        for w in list(vocab_i):
            d[w] += 1

    df_idf = pd.DataFrame()
    df_idf['count'] = d.values()
    df_idf['word'] = d.keys()
    df_idf['idf'] = np.log(len(documents) / df_idf['count'])
    return df_idf


def create_document_vector(documents, w2t, n_embedding):
    """
    学習済みの word topic vector と分かち書き済みの文章, 使用されている単語から
    文章ベクトルを作成するメソッド.
    Args:
        documents(list[list[str]]):
        w2t(dict): 単語 -> 埋め込み次元の dict
        n_embedding(int):
    Returns:
        embedded document vector
    """
    doc_vectors = []

    for doc in documents:
        vector_i = np.zeros(shape=(n_embedding,))
        for w in doc:
            try:
                v = w2t[w]
                vector_i += v
            except KeyError:
                continue
        doc_vectors.append(vector_i)
    return np.array(doc_vectors)


def compress_document_vector(doc_vector, p=.04):
    v = np.copy(doc_vector)
    vec_norm = np.linalg.norm(v, axis=1)
    # zero divide しないように
    vec_norm = np.where(vec_norm > 0, vec_norm, 1.)
    v /= vec_norm[:, None]

    a_min = v.min(axis=1).mean()
    a_max = v.max(axis=1).mean()
    threshold = (abs(a_min) + abs(a_max)) / 2. * p
    v[abs(v) < threshold] = .0
    return v


def download_from_gdrive(id, destination):
    """
    Download file from Google Drive
    :param str id: g-drive id
    :param str destination: output path
    :return:
    """
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        print("get download warning. set confirm token.")
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    """
    verify whether warned or not.
    [note] In Google Drive Api, if requests content size is large,
    the user are send to verification page.
    :param requests.Response response:
    :return:
    """
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            return v

    return None


def save_response_content(response, destination):
    """
    :param requests.Response response:
    :param str destination:
    :return:
    """
    chunk_size = 1024 * 1024
    print("start downloading...")
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), unit="MB"):
            f.write(chunk)
    print("Finish!!")
    print("Save to:{}".format(destination))


def ja_word_vector(output_dir):
    """
    fasttext を用いて学習された日本語の word vector を取得します
    データは以下の記事にあるものを使わせてもらっています. 感謝してつかいましょう^_^
    > https://qiita.com/Hironsan/items/513b9f93752ecee9e670
    Args:
        output_dir: 保存先ディレクトリ
    Returns:
    """
    file_name = 'vector_neologd'
    dl_path = os.path.join(output_dir, '{}.zip'.format(file_name))
    # 展開すると model.vec という名前のファイルがあるのでそれが本体
    model_path = os.path.join(output_dir, 'model.vec')

    if not os.path.exists(model_path):
        os.makedirs(output_dir, exist_ok=True)
        download_from_gdrive('0ByFQ96A4DgSPUm9wVWRLdm5qbmc', destination=dl_path)
        with zipfile.ZipFile(dl_path) as f:
            f.extractall(output_dir)
    else:
        print('model already exist')

    m = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
    return m


def create_scdv_vector(parsed_docs, word_vec, n_components, output_dir):
    n_wv_embed = word_vec.vector_size

    # w2v model と corpus の語彙集合を作成
    vocab_model = set(k for k in word_vec.vocab.keys())
    vocab_docs = set([w for doc in parsed_docs for w in doc])
    out_of_vocabs = len(vocab_docs) - len(vocab_docs & vocab_model)
    print('out of vocabs: {out_of_vocabs}'.format(**locals()))

    # 使う文章に入っているものだけ学習させるため共通集合を取得してその word vector を GMM の入力にする
    use_words = list(vocab_docs & vocab_model)

    # はじめに文章全体の idf を作成した後, use_word だけの df と left join して
    # 使用している単語の idf を取得
    df_use = pd.DataFrame()
    df_use['word'] = use_words
    df_idf = create_idf_dataframe(parsed_docs)
    df_use = pd.merge(df_use, df_idf, on='word', how='left')
    idf = df_use['idf'].values

    # 使う単語分だけ word vector を取得. よって shape = (n_vocabs, n_wv_embed,)
    use_word_vectors = np.array([word_vec[w] for w in use_words])

    # 公式実装: https://github.com/dheeraj7596/SCDV/blob/master/20news/SCDV.py#L32 により tied で学習
    # 共分散行列全部推定する必要が有るほど低次元ではないという判断?
    # -> 多分各クラスの分散を共通化することで各クラスに所属するデータ数を揃えたいとうのがお気持ちっぽい
    clf = GaussianMixture(n_components=n_components, covariance_type='tied', verbose=2)
    clf.fit(use_word_vectors)

    # word probs は各単語のクラスタへの割当確率なので shape = (n_vocabs, n_components,)
    word_probs = clf.predict_proba(use_word_vectors)

    # 単語ごとにクラスタへの割当確率を wv に対して掛け算する
    # shape = (n_vocabs, n_components, n_wv_embed) になる
    word_cluster_vector = use_word_vectors[:, None, :] * word_probs[:, :, None]

    # topic vector を計算するときに concatenation するとあるが
    # 単に 二次元のベクトルに変形して各 vocab に対して idf をかければ OK
    topic_vector = word_cluster_vector.reshape(-1, n_components * n_wv_embed) * idf[:, None]
    # nanで影響が出ないように 0 で埋める
    topic_vector[np.isnan(topic_vector)] = 0
    word_to_topic = dict(zip(use_words, topic_vector))

    np.save(os.path.join(output_dir, 'word_topic_vector.npy'), topic_vector)

    topic_vector = np.load(os.path.join(output_dir, 'word_topic_vector.npy'))
    n_embedding = topic_vector.shape[1]

    cdv_vector = create_document_vector(parsed_docs, word_to_topic, n_embedding)
    np.save(os.path.join(output_dir, 'raw_document_vector.npy'), cdv_vector)

    compressed = compress_document_vector(cdv_vector)
    np.save(os.path.join(output_dir, 'compressed_document_vector.npy'), compressed)

    return compressed


def create_swem(doc, word_vec, aggregation='max'):
    """
    Create Simple Word Embedding Model Vector from document (i.e. list of sentence)
    Args:
        doc(list[list[str]]):
        aggregation(str): `"max"` or `"mean"`
    Returns:
    """
    print('create SWEM: {}'.format(aggregation))
    if aggregation == 'max':
        agg = np.max
    elif aggregation == 'mean':
        agg = np.mean
    else:
        raise ValueError()

    swem = []
    for sentence in tqdm(doc, total=len(doc)):
        embed_i = [convert_to_wv(s, word_vec) for s in sentence]
        embed_i = np.array(embed_i)

        # max-pooling で各文章を 300 次元のベクトルに圧縮する
        embed_i = agg(embed_i, axis=0)
        swem.append(embed_i)
    swem = np.array(swem)
    return swem


def convert_to_wv(w, word_vec):
    """
    単語から word2vec の特徴量に変換する.
    単語が登録されていない (vocabularyにない) ときは zero vector を返す

    args:
        w(str): 変換したい word
    """
    try:
        v = word_vec.word_vec(w)
    except KeyError:
        # print(e)
        v = np.zeros(shape=(word_vec.vector_size,))
    return v


def create_n_gram_feature(docs, n_gram=5):
    vectors = []
    for sentence in tqdm(docs, total=len(docs)):
        v = create_n_gram_max_pooling(sentence, n_gram)
        vectors.append(v)
    return np.array(vectors)


# N-Gram SWEM
# n-gram の平均ベクトルを用いて max-pooling する swem の発展形
def create_n_gram_max_pooling(s, word_vec, n_gram_length=5):
    """
    単語に区切られた文字列から n_gram の max-pooling vector を返す

    Returns:
        np.ndarray: shape = (n_embedded_dim, )
    """
    embed_i = [convert_to_wv(w, word_vec) for w in s]
    gram_vectors = []
    for i in range(max(1, len(embed_i) - n_gram_length)):
        gram_i = embed_i[i:i + n_gram_length]
        gram_mean = np.mean(gram_i, axis=0)
        gram_vectors.append(gram_mean)
    return np.max(gram_vectors, axis=0)


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
