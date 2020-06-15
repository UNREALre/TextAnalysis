#! /usr/bin/env python
# -*- coding: utf-8 -*-

from nltk import (pos_tag, sent_tokenize, wordpunct_tokenize)
from classes.CustomCorpusReader import HTMLCorpusReader
import os
import pickle


class Preprocessor(object):

    """Обёртка над HTMLCorpusReader"""

    def __init__(self, corpus, target, **kwargs):
        self.corpus = corpus
        self.target = target

    def fileids(self, fileids=None, categories=None):
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def abspath(self, fileid):
        # Ищем путь к каталогу относительно корня исходного корпуса
        parent = os.path.relpath(os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root)

        # Выделяем части пути для реконструирования
        basename = os.path.basename(fileid)
        name, ext = os.path.splitext(basename)

        # Создаем имя файла с расширением .pickle
        basename = name + '.pickle'

        # Возвращаем путь к файлу относительно корня целевого корпуса
        return os.path.normpath(os.path.join(self.target, parent, basename))

    def tokenize(self, fileid):
        for paragraph in self.corpus.paras(fileids=fileid):
            yield[
                pos_tag(wordpunct_tokenize(sent), lang='rus')
                for sent in sent_tokenize(paragraph)
            ]

    def process(self, fileid):
        """
        Записывает трансформированный документ в виде сжатого архива в заданное место.

        Вызывается для одного файла, проверяет местоположение на диске, чтобы избежать ошибок.
        Использует tokenize() для предварительной обработки. Полученные данные и записываются в файл.
        """

        # Определяем путь к файлу для записи результата
        target = self.abspath(fileid)
        parent = os.path.dirname(target)

        # Убеждаемся, что каталог существует
        if not os.path.exists(parent):
            os.makedirs(parent)

        # Убеждаемся, что parent - папку, а не файл
        if not os.path.isdir(parent):
            raise ValueError("Нужно предоставить папку для записи обработанных данных!")

        # Создаем структуру данных для записи в архив
        document = list(self.tokenize(fileid))

        # Пишем данные в архив на диск
        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

        # Удаляем документ из памяти
        del document

        # Возвращаем путь к целевому файлу
        return target

    def transform(self, fileids=None, categories=None):
        """Метод, вызывающий process()"""

        # Создаем целевой каталог, если он еще не создан
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # Получить имена файлов для обработки
        for fileid in self.fileids(fileids, categories):
            yield self.process(fileid)
