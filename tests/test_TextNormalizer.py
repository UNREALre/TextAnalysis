#! /usr/bin/env python
# -*- coding: utf-8 -*-

from classes.PickledCorpusReader import PickledCorpusReader
from classes.TextNormalizer import TextNormalizer
from config import CORPUS_PREPROC_ROOT

pickled_reader = PickledCorpusReader(CORPUS_PREPROC_ROOT)
text_normalizer = TextNormalizer()


print('Тексты документов 3 и 5, возвращается генератор')
texts = pickled_reader.docs(['Category 2/document3.pickle', 'Category N/document5.pickle'])
print('Normalization ... ')
normalized_texts = text_normalizer.transform(texts)
for normalized_text in normalized_texts:
    print(normalized_text)
