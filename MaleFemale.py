#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модель, использующая лингвистические признаки для выявления преобладающего рода во фрагменте текста
"""

from collections import Counter
import nltk
import urllib
from bs4 import BeautifulSoup

# Классификаторы предложений
MALE = 'male'  # речь только о мужчинах идет
FEMALE = 'female'  # речь только о женщинах идет
UNKNOWN = 'unknown'  # непонятно о чем речь
BOTH = 'both'  # речь и о мужчинах, и о женщинах

MALE_WORDS = {'он', 'мальчик', 'парень', 'мужчина', 'парнишка', 'пацан', 'мужик', 'мужичок', 'друг', 'муж', 'любовник',
              'брат', 'сын', 'дядя', 'племянник', 'двоюродный брат', 'отец', 'папа', 'чувак', 'джентельмен', 'господин',
              'товарищ', 'дедушка', 'дед', 'солдат', 'боец', 'служивый', 'король', 'мистер', 'священник', 'принц',
              'шейх', 'официант', 'основатель', 'учитель', 'насильник', 'ученик'}

FEMALE_WORDS = {'она', 'девочка', 'девушка', 'женщина', 'жена', 'дочка', 'сестра', 'мама', 'племянница',
                'бабушка', 'тётя', 'любовница', 'подруга', 'подружка', 'свекровь', 'богиня', 'королева',
                'принцесса', 'леди', 'госпожа', 'официантка', 'стюардесса', 'вдова', 'проститутка',
                'мисс', 'миссис', 'красотка', 'очаровашка', 'милашка', 'основательница', 'учительница',
                'ученица'}


def genderize(words):
    """
    Принимает на вход предложение из слов.
    Подсчитывает кол-во вхождений слов из вышезаданных классификаторов в предложении.
    Возвращает  классификатор предложения
    """
    male_words_len = len(MALE_WORDS.intersection(words))
    female_words_len = len(FEMALE_WORDS.intersection(words))

    if male_words_len > 0 and female_words_len == 0:
        return MALE
    elif male_words_len == 0 and female_words_len > 0:
        return FEMALE
    elif male_words_len > 0 and female_words_len > 0:
        return BOTH
    else:
        return UNKNOWN


def count_gender(sentences):
    """
    Принимает на вход список предложений.
    Использует genderize для классификации предложения и для определения общего кол-ва слов признаков.
    Возвращает список классифицируемый по каждому предложению (каким оно является) и возвращает
    количество слов в каждом из предложений списка
    """
    sents = Counter()
    words = Counter()

    for sentence in sentences:
        gender = genderize(sentence)
        sents[gender] += 1
        words[gender] += len(sentences)

    return sents, words


def parse_gender(text):
    """
    Принимает текст статьи
    Выводит результат текстового анализа
    """
    sentences = [
        [word.lower() for word in nltk.word_tokenize(sentence)]
        for sentence in nltk.sent_tokenize(text)
    ]

    sents, words = count_gender(sentences)
    total = sum(words.values())

    for gender, count in words.items():
        pcent = round((count / total) * 100, 2)
        nsents = round(sents[gender], 2)

        print('%s %s (%s sentences)' % (pcent, gender, nsents))


def get_esquire_article():
    """
    Спаршивает статью
    """
    url = "https://esquire.ru/articles/181453-kto-takoy-viktor-stolbun-i-kak-emu-udalos-osnovat-vliyatelnuyu-sektu-v-kotoroy-sostoyal-eduard-uspenskiy-i-gde-praktikovali-nasilie-i-lzhenauku/#part1"
    page = urllib.request.urlopen(url).read().decode("utf-8")
    soup = BeautifulSoup(page, "html.parser")
    text = soup.find("div", {"class": "text-page"}).text

    return text
