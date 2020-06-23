#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Приемы векторизации в Gensim представляют особый интерес, т.к. библиотека позволяет сохранять и загружать
корпусы с диска, отделяя их от конвейера обработки.

Текущий преобразователь будет обертывать объект Dictionary из Gensim, сгенерированный методом fit, чей метод doc2bow
используется методом transform().
Объект Dictionary (например, TfidfModel) можно сохранить на диск и загрузить с диска, поэтому текущий преобразователь
тоже будет пользоваться такой возможностью. Путь сохранения будет определяться при создании экземпляра (инит).
"""

import os
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from sklearn.base import BaseEstimator, TransformerMixin


class GensimVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, path=None):
        self.path = path
        self.id2word = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            self.id2word = Dictionary.load(self.path)

    def save(self):
        self.id2word.save(self.path)

    def fit(self, documents, labels=None):
        """
        Конструирует объект Dictionary, передавая его конструктору лексемизированные и нормализованные документы.

        Экземпляр сразу сохраняется на диск для последующей загрузки без повторного обучения.
        """

        self.id2word = Dictionary(documents)
        self.save()
        return self

    def transform(self, documents):
        """
        Вызывает метод Dictionary.doc2bow, возвращающий разреженное представление документа в виде списка
        кортежей (token_id, frequency). Чтобы такое представление не вызывало проблем у Scikit-Learn, используется
        далее sparse2full из Gensim для преобразования полученного представления в массив NumPy.
        """

        for document in documents:
            docvec = self.id2word.doc2bow(document)
            yield sparse2full(docvec, len(self.id2word))
