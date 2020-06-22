#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Учимся применять Scikit-Learn API

Основой Scikit-Learn - является "API для машинного обучения", предоставляющий доступ к реализациям
широкого спектра моделей через единый интерфейс. Благодаря этому, можно использовать библиотеку для
одновременного обучения большого количества моделей и сравнения предсказаний этих моделей.

API - объектно-ориентированный и определяется иерархией интерфейсов для разных задач машинного обучения.
Корень иерархии - Estimator - любой объект, способный обучаться на данных. Т.е. по сути Estimator - играет
роль интерфейса, а класс, что описан ниже - его реализация.

Т.к. класс наследует BaseEstimator, то можно использовать метод fit_predict, который объединяет fit и
predict в один вызов.

Ниже просто пример определения интерфейса.
"""

from sklearn.base import BaseEstimator


class Estimator(BaseEstimator):

    """Класс, реализующий два основных метода - fit и predict"""
    def fit(self, X, y = None):
        """
        Настраивает состояние экземпляра Estimator, опираясь на данные для обучения X и y.

        Обучающие данные X должны иметь форму матрицы. В процессе обучения внутреннее состояние
        экземпляра Estimator изменяется так, чтобы подготовить его к предсказаниям. Т.к. метод изменяет
        внутреннее состояние, он возвращает ссылку self на сам экземпляр, что позволяет составлять цепочки
        из вызовов методов.
        """

        return self

    def predict(self, X):
        """
        Возвращает вектор предсказаний для каждой строки

        Создает предсказания, используя внутренее состояние модули, обученной на данных X.
        На вход подается матрица с тем же кол-вом столбцов, что было в обучающей, которую передавали в fit.
        Возвращает вектор yhat, содержащий предсказания для всех строк во входных данных.
        """

        return yhat
