#! /usr/bin/env python
# -*- coding: utf-8 -*-

from classes.CustomCorpusReader import HTMLCorpusReader
from config import CORPUS_ROOT

html_reader = HTMLCorpusReader(CORPUS_ROOT)
print('Документы корпуса', html_reader.fileids())
print('Категории корпуса', html_reader.categories())
print('Документы категории Category 1 и Category N корпуса', html_reader.resolve(None, ['Category 1', 'Category N']))
print('Документы, которые нужны явно указанные', html_reader.resolve(['document3.txt', 'document1.txt', 'some123'], None))

print()
print()

print('Текущее состояние корпуса:\n{}'.format(html_reader.describe()))

print()
print()

"""
print('Тексты документов 3 и 5, возвращается генератор')
texts = html_reader.docs(['Category 2/document3.txt', 'Category N/document5.txt'])
for text in texts:
    print(text)

print()
print()

print('Очищенные readability тексты html документов - возвращается генератор')
cleaned_texts = html_reader.html(['Category 1/document2.txt'])
for cleaned_text in cleaned_texts:
    print(cleaned_text)

print()
print()

print('Выделенные параграфы документа')
paras = html_reader.paras(['Category 1/document2.txt'])
for current_paras in paras:
    print("__{}__".format(current_paras))

print()
print()

print('Выделенные предложения из параграфов документа')
sents = html_reader.sents(['Category 1/document2.txt'])
for sent in sents:
    print("_{}_".format(sent))

print()
print()

print('Выделенные слова из предложений')
words = html_reader.words(['Category 1/document2.txt'])
for word in words:
    print("~{}~".format(word))

print()
print()
"""

print('Маркировка лексем в документе кортежем (тег, токен)')
tokens = html_reader.tokenize(['Category 1/document2.txt'])
for token in tokens:
    print(token)
