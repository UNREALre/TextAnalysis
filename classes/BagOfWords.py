#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Мешок слов - эксперименты с векторизацией

Реализуется векторизация с помощью NLTK, Scikit и Gensim. Можно увидеть разницу в методах.

Типы реализованных кодирований (векторизаций):
1. Частотная - когда текст разбивается на лексемы, а затем составляется вектор, каждый элемент которого
обозначает количество вхождений текущей лексемы документа в словарь лексем.

2. Прямое кодирование (логическая векторизация) - текст так же разбивается на лексемы, а затем
составляется вектор, каждый элемент которого содержит либо True, либо False, в зависимости от того,
входит ли текущая лексема документа в словарь или нет.
Прямое кодирование - хорошо используется для определения сходства и различия на уровне документов(!).
Т.к. все слова получаются равноудаленными в этом методе, то он не в состоянии закодировать сходство на
уровне слов.

3. Частота слова - обратная частота документа (метод кодирования TF-IDF - Term Frequency-Inverse Document
Frequency) - основывается на сопоставлении относительной частоты лексем в документе с их частотой в других
документах корпуса. Этот метод придает больше веса терминам, релевантным для конкретного экземпляра (чем реже
лексема текущего документа встречается в других документах, тем больше вес) - т.е. эти лексемы наиболее
хорошо характеризуют текущий документ.
Частота слова для заданной лексемы в документе tf(t,d) может быть логической частотой (1, если есть или 0),
но обычно частота слова и обратнаячастота документа масштабируются логарифмически, чтобы избежать
предвзятости в пользую наиболее длинных документов или слов, появляющихся особенно часто. Поэтому, оценка
такая: tf(t,d) = 1 + log(f(t,d))
Обратную частоту документа для слова в заданном наборе документов вычисляют так:
idf(t,D) = log1 + N/nt, где N - число документов, а nt - число вхождений слова t во всех документах.
Итоговая полная оценка TF-IDF:
tfidf(t,d,D) = tf(t,d) * idf(t,D)
С помощью TF-IDF легко отлавливать стоп-слова, т.е. ничего не значащие слова (типа предлогов и союзов),
которые встречаются везде.

Лучшее кодирование: TF-IDF
"""

import nltk
import string
import gensim
from collections import defaultdict
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from sklearn.preprocessing import Binarizer
from nltk.text import TextCollection


def tokenize(text):
    """
    Упрощенная лексимизация

    Отбрасывает знаки препинания, используя набор символов string.puctuation.
    Преобразует оставшиеся символы в нижний регистр.
    Сворачивает свойства с помощью SnowballStemmer (по типу удаления суффиксов мн.числа в en и тд)
    :param text:
    :return:
    """

    stem = nltk.stem.SnowballStemmer('russian')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation:
            continue

        yield stem.stem(token)


def nltk_vectorize(doc):
    """Реализует базовую частотную векторизацию (используем NLTK)"""

    # defaultdict позволяет указать, какое значение должен вернуть словарь, при обращении
    # к несуществующему ключу. Вызывая defaultdict(int) указываем, что возвращать надо 0
    features = defaultdict(int)
    for token in tokenize(doc):
        features[token] += 1

    return features


def nltk_logical_vectorize(doc):
    """
    Реализует логическую векторизацию (прямое кодирование, NLTK)

    Логическая векторизация подходит для коротких предложений/твитов. Она не учитывает частоту,
    просто помещает в соотв-щий эл-нт вектор значение True, если лексема есть в документе, False - иначе
    """

    doc = tokenize(doc)
    # не обязательно возвращать False, возвращаем лишь True для найденных лексем
    return {
        token: True
        for token in doc
    }


def nltk_tfidf_vectorize(corpus):
    """
    Реализует TF-IDF кодирование с помощью NLTK

    Выполняет обход всех документов и возвращает словарь, ключами которого являются слова,
    а значениями - их оценки TF-IDF для данного конкретного документа
    """

    corpus = [
        [token for token in tokenize(doc)]
        for doc in corpus
    ]

    # TextCollection - обертывает список документов или корпус. Поддерживает подсчет вхождений,
    # вычисление оценки согласованности (конкордантности), определение устойчивых словосочетаний и
    # непосредственно то, что нам надо - вычисляет tf_idf
    texts = TextCollection(corpus)

    for doc in corpus:
        yield {
            term: round(texts.tf_idf(term, doc), 3)
            for term in doc
        }


def scikit_vectorize(corpus):
    """Реализует базовую частотную векторизацию (используем Scikit)"""

    vectorizer = CountVectorizer()
    # fit - принимает итерируемую последовательность и создает словарь корпуса
    # transform - преобразует каждый элемент последовательность в разреженный массив, индексы в котором -
    # кортежи с идентификаторами документов и лексем из словаря, а значения - счетчики лексем
    vectors = vectorizer.fit_transform(corpus)
    print('Scikit vocabulary: {}'.format(vectorizer.vocabulary_))
    return vectors


def scikit_logical_vectorize(corpus):
    """Реализует логическую векторизацию (прямое кодирование, Scikit)"""

    # Т.к. Binarizer принимает только числовые данные, то сначала преобразуем в численное пространство текст
    freq = CountVectorizer()
    corpus = freq.fit_transform(corpus)

    # Пропущенные через Binarizer элементы вектора, имеющие пороговое значение <= 0 (в нашем случае 0),
    # останутся нулем, остальные получат значение 1 (то что и надо нам - True / False)
    onehot = Binarizer()
    corpus = onehot.fit_transform(corpus.toarray())

    return corpus


def scikit_tfidf_vectorize(corpus):
    """Реализует TF-IDF кодирование с помощью Scikit"""

    tfidf = TfidfVectorizer()
    # Результатом работы преобразователя является разреженная матрица вида ((doc,term), tfidf), где каждый
    # ключ является парой "документ/лексема", а значение - оценка TF-IDF
    corpus = tfidf.fit_transform(corpus)

    return corpus


def gensim_vectorize(corpus):
    """Реализует базовую частотную векторизацию (используем Gensim)"""

    tokenized_corpus = [tokenize(doc) for doc in corpus]
    # Создаем объект словаря Dictionary отображающий лексемы в индексы в порядке их следования в документе
    # т.е. в id2word будет словарь документа корпуса - ['грил', 'для', 'жарк', 'и', 'идеальн']
    id2word = gensim.corpora.Dictionary(tokenized_corpus)

    # повторно создаем генератор, т.е. id2word его израсходовал
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    # doc2bow - принимает лексимизированный документ и возвращает матрицу кортежей (id,count), где id -
    # идентификатор лексемы в словаре
    vectors = [id2word.doc2bow(doc) for doc in tokenized_corpus]

    return vectors


def gensim_logical_vectorize(corpus):
    """Реализует логическую векторизацию (прямое кодирование, Gensim)"""

    tokenized_corpus = [tokenize(doc) for doc in corpus]
    id2word = gensim.corpora.Dictionary(tokenized_corpus)

    tokenized_corpus = [tokenize(doc) for doc in corpus]
    vectors = [
        [(token[0], 1) for token in id2word.doc2bow(doc)]
        for doc in tokenized_corpus
    ]

    return vectors


def gensim_tfidf_vectorize(corpus):
    """Реализует TF-IDF кодирование с помощью Gensum"""

    tokenized_corpus = [tokenize(doc) for doc in corpus]
    lexicon = gensim.corpora.Dictionary(tokenized_corpus)
    tfidf = gensim.models.TfidfModel(dictionary=lexicon, normalize=True)

    tokenized_corpus = [tokenize(doc) for doc in corpus]
    vectors = [tfidf[lexicon.doc2bow(doc)] for doc in tokenized_corpus]

    return vectors


def print_vectors(vectors):
    for vector in vectors:
        print(vector)
        print()


corpus = [
    "Идеально для оладий подходят молоденькие кабачки и цуккини. Для жарки кабачков используем масло и гриль.",
    "В сковороду выложить фарш и обжарить на среднем огне, до готовности, постоянно разминая деревянной лопаткой, чтобы разбить крупные комки.",
    "Вкусный и полезный ужин из запеченных кабачков с фаршем, помидорами и сыром, любителям жаркого!"
]

types = {
    'NLTK vectorizer': {
        'frequency type': nltk_vectorize,
        'logical type': nltk_logical_vectorize,
        'TF-IDF': nltk_tfidf_vectorize

    },
    'Scikit-Learn vectorizer': {
        'frequency type': scikit_vectorize,
        'logical type': scikit_logical_vectorize,
        'TF-IDF': scikit_tfidf_vectorize
    },
    'Gensim vectorizer': {
        'frequency type': gensim_vectorize,
        'logical_type': gensim_logical_vectorize,
        'TF-IDF': gensim_tfidf_vectorize
    }
}


# Тестируем написанные выше методы векторизации
for vectorizer_name, vectorizers in types.items():
    print(vectorizer_name)
    for vectorizer_type, vectorizer_func in vectorizers.items():
        print(vectorizer_type)
        if vectorizer_name == 'NLTK vectorizer' and vectorizer_type != 'TF-IDF':
            # Применяем функцию векторизации ко всем документам в корпусе с помощью map
            vectors = map(vectorizer_func, corpus)
            print_vectors(vectors)
        else:
            vectors = vectorizer_func(corpus)
            print_vectors(vectors)
