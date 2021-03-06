#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
В примере выбирается семейство моделей наивного байесовского классификатора и конкретного члена
семейства - полиномиальную модель (которая подходит для классификации текста).
Модель определяется при создании класса вместе с гиперпараметрами:
alpha - для аддитивного сглаживания;
спиок - предварительные вероятности для каждого из двух классов

Модель должна обучиться на указанных данных (documents и labels) и превратиться в обученную модель.

"""

from sklearn.naive_bayes import MultinomialNB
from classes import Estimator

model = MultinomialNB(alpha=0.0, class_prior=[0.4, 0.6])
model.fit(documents, labels)