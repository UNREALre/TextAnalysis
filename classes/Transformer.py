#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer - это специальный тип Estimator-а, создающий новый набор данных из старых, опираясь на правила,
выявленные в процессе обучения.

Пример интерфейса ниже
"""

from sklearn.base import TransformerMixin, BaseEstimator


class Transformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        """Изучает правила преобразования на основе входных данных X"""

        return self

    def transform(self, X):
        """Преобразует X в новый набор данных Xprime и возвращает его"""

        return Xprime

