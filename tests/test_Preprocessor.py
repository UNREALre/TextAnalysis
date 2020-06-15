#! /usr/bin/env python
# -*- coding: utf-8 -*-

from classes.Preprocessor import Preprocessor
from classes.CustomCorpusReader import HTMLCorpusReader
from config import CORPUS_ROOT, CORPUS_PREPROC_ROOT

corpus = HTMLCorpusReader(CORPUS_ROOT)
preprocessor = Preprocessor(corpus, CORPUS_PREPROC_ROOT)
for transform in preprocessor.transform(['Category 1/document2.txt']):
    print(transform)