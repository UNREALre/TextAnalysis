#! /usr/bin/env python
# -*- coding: utf-8 -*-

from classes.PickledCorpusReader import PickledCorpusReader
from config import CORPUS_PREPROC_ROOT

pickled_reader = PickledCorpusReader(CORPUS_PREPROC_ROOT)

print('Документы корпуса', pickled_reader.fileids())
print('Категории корпуса', pickled_reader.categories())
print('Документы категории Category 1 и Category N корпуса', pickled_reader.resolve(None, ['Category 1', 'Category N']))
print('Документы, которые нужны явно указанные', pickled_reader.resolve(['document3.pickle', 'document1.pickle', 'some123'], None))

print()
print()


print('Тексты документов 3 и 5, возвращается генератор')
texts = pickled_reader.docs(['Category 2/document3.pickle', 'Category N/document5.pickle'])
for text in texts:
    print(text)

print()
print()

print('Выделенные параграфы документа')
paras = pickled_reader.paras(['Category 1/document2.pickle'])
for current_paras in paras:
    print("__{}__".format(current_paras))

print()
print()

print('Выделенные предложения из параграфов документа')
sents = pickled_reader.sents(['Category 1/document2.pickle'])
for sent in sents:
    print("_{}_".format(sent))

print()
print()

print('Выделенные слова из предложений')
words = pickled_reader.words(['Category 1/document2.pickle'])
for word in words:
    print("~{}~".format(word))

print()
print()

print('Выделенные кортежи лексем и маркеров из предложений')
tagged = pickled_reader.tagged(['Category 1/document2.pickle'])
for word, tag in tagged:
    print("{} = {}".format(word, tag))

print()
print()
