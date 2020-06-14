#! /usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from readability.readability import (Unparseable, Document as Paper)
import codecs
import os

CAT_PATTERN = r'([\w_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.txt'
TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']


class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    Объект чтения корпуса с HTML-документами для получения дополнительных
    возможностей предварительной обработки данных
    """

    def __init__(self, root, fileids=DOC_PATTERN, encoding='utf-8', tags=TAGS, **kwargs):
        """
        Инициализирует объект чтения корпуса.
        """

        # Если шаблон категорий не был передан в класс явно - добавляем его
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        # Инициализируем объекты чтения корпуса из NLTK
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)

        # Сохранить теги, подлежащие извлечению
        self.tags = tags

    def resolve(self, fileids, categories):
        """
        Фильтрация файлов корпуса на диске.
        """

        if fileids is not None and categories is not None:
            raise ValueError("Укажите fileids или categories, но не то и другое разом")

        if categories is not None:
            # вызываем метод CorpusReader-а, возвращающего файлы переданной категории
            return self.fileids(categories)

        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Возвращает полный текст документа
        """

        # Получаем список файлов
        fileids = self.resolve(fileids, categories)

        # Создаем генератор, загружающий документы в память по одному
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield f.read()

    def sizes(self, fileids=None, categories=None):
        """
        Возвращает список кортежей: идентификатор файла и его размер.
        Помогает выявить чрезмерно большие файлы, которых быть не должно
        """

        fileids = self.resolve(fileids, categories)

        # Создаем генератор, возвращающий имена и размеры файлов
        for path in self.abspaths(fileids):
            yield path, os.path.getsize(path)

    def html(self, fileids=None, categories=None):
        """Возвращает содержимое HTML каждого документа, очищая его с помощью readability"""
        for doc in self.docs(fileids, categories):
            try:
                yield Paper(doc).summary()
            except Unparseable as e:
                print("Невозможно распарсить HTML: {}".format(e))
                continue
