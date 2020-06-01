#! /usr/bin/env python
# -*- coding: utf-8 -*-

from classes.CustomCorpusReader import HTMLCorpusReader
from config import CORPUS_ROOT

html_reader = HTMLCorpusReader(CORPUS_ROOT)
print('Документы корпуса', html_reader.fileids())
print('Категории корпуса', html_reader.categories())
print('Документы категории Category 1 и Category N корпуса', html_reader.resolve(None, ['Category 1', 'Category N']))
print('Документы, которые нужны явно указанные', html_reader.resolve(['document3.txt', 'document1.txt', 'some123'], None))
