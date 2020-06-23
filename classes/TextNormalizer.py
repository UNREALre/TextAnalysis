#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Основные методы нормализации текста:

1. Фильтрация лексем.
Удаление знаков препинания и стоп-слов.

2. Стемминг.
Используя серию правил разделяет строки на меньшие подстроки для того, чтобы в последующем удалить
приставки, суффиксы и вообще все лишнее из слов, меняющих их занчение.
Пример:
"Идеально для оладий подходят молоденькие кабачки и цуккини."
{'идеальн', 'для', 'олад', 'подход', 'молоденьк', 'кабачк', 'и', 'цуккин'}

3. Лемматизация.
Ищет каждую лексемму в словаре и возвращает каноническую (словарную) форму слова, которая и называется
леммой.
Пример:
gardening -> to garden
garden -> garden
gardener -> gardener

Все это разные леммы для метода лемматизации, в то время как для стемминга - все это было бы одной
лексемой - garden.

Ниже представлен преобразователь, объединяющий все эти методы.

P.S. К другим методам преобразования текста для последующих обучений можно отнести:
- удаление лексем, появляющихся чаще или реже какого-либо предела
- удаление стоп-слов с последующим выбором первых 5-10 тысяч наиболее встречающихся слов
- вычисление накопленной частоты с последующим отбором слов, составляющих 10-50% накопленной частоты.

Все вышеописанное позволяет игнорировать очень редкие и очень частые слова в тексте, выявляя при этом
термины, потенциально наиболее значимые для прогнозирования в данном корпусе.
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('stopwords')
nltk.download('wordnet')


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='russian'):
        """Принимает на вход язык, используемый для загрузки правильного набора стоп-слов из NLTK."""

        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        """Сравнивает первую букву в названии категории Юникода каждого символа с P (Punctuation)"""

        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token):
        """Проверяет, присутствут ли данная лекса в множестве стоп-слов"""

        return token.lower() in self.stopwords

    def normalize(self, document):
        """
        Применяет функции фильтрации для удаления нежелательных лексем и выполняет лемматиизацию

        Принимает document, являющиеся списком абзацев, состоящих из списка предложений, которые
        представлены списками кортежей (token, tag)
        """

        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
        """
        Преобразует теги частей речи из набора Penn Treebank, используемый фун-ей nltk.pos_tag,
        в теги WordNet, выбирая по умолчению существительное.
        """

        tag = {
            'S': wn.NOUN,
            'V': wn.VERB,
            'ADV': wn.ADV,
            'A': wn.ADJ
        }.get(pos_tag, wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            print(document)
            yield self.normalize(document)
