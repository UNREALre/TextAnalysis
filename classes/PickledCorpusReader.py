#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from nltk.corpus.reader.api import (CorpusReader, CategorizedCorpusReader)
from classes.CustomCorpusReader import HTMLCorpusReader, CAT_PATTERN

PKL_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.pickle'


class PickledCorpusReader(HTMLCorpusReader):

    """Класс наследует HTMLCorpusReader, но работает не с исходным корпусом, а с обработанным препроцессором"""

    def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def docs(self, fileids=None, categories=None):
        """Переопределенный docs из HTMLCorpusReader - загружает документы из архивов"""

        fileids = self.resolve(fileids, categories)

        # Загружаем документы в память по одному
        for path in self.abspaths(fileids):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def paras(self, fileids=None, categories=None):
        """Переопределяем paras, потому что документ, прошедший обработку, хранится как список абзацев"""

        for doc in self.docs(fileids, categories):
            for para in doc:
                yield para

    def sents(self, fileids=None, categories=None):
        """Переопределяем sents, т.к. каждый абзац теперь - список предложений"""

        for para in self.paras(fileids, categories):
            for sent in para:
                yield sent

    def tagged(self, fileids=None, categories=None):
        """Т.к. предложение у нас список кортежей лексем и тегов, то возвращает лексемы с тегами"""

        for sent in self.sents(fileids, categories):
            for tagged_token in sent:
                yield tagged_token

    def words(self, fileids=None, categories=None):
        """Переопределяем words, т.к. предложение у нас теперт - это список кортежей лексем и тегов"""

        for tagged in self.tagged(fileids, categories):
            yield tagged[0]
