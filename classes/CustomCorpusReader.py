#! /usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.corpus.reader.api import (CorpusReader, CategorizedCorpusReader)
from nltk import (sent_tokenize, wordpunct_tokenize, pos_tag, download)
download('averaged_perceptron_tagger')
download('averaged_perceptron_tagger_ru')
from readability.readability import (Unparseable, Document as Paper)
import codecs
import os
import bs4
import re

CAT_PATTERN = r'([\w_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.txt'
TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']  # теги для извлечения абзацев


class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    Объект чтения корпуса с HTML-документами для получения дополнительных
    возможностей предварительной обработки данных
    """

    def __init__(self, root, fileids=DOC_PATTERN, encoding='utf-8', tags=TAGS, **kwargs):
        """Инициализирует объект чтения корпуса."""

        # Если шаблон категорий не был передан в класс явно - добавляем его
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        # Инициализируем объекты чтения корпуса из NLTK
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)

        # Сохранить теги, подлежащие извлечению
        self.tags = tags

    def resolve(self, fileids, categories):
        """Фильтрация файлов корпуса на диске."""

        if fileids is not None and categories is not None:
            raise ValueError("Укажите fileids или categories, но не то и другое разом")

        if categories is not None:
            # вызываем метод CorpusReader-а, возвращающего файлы переданной категории
            return self.fileids(categories)

        return fileids

    def docs(self, fileids=None, categories=None):
        """Возвращает полный текст документа"""

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
        """Возвращает содержимое HTML каждого документа, очищая его с помощью readability."""

        for doc in self.docs(fileids, categories):
            try:
                yield Paper(doc).summary()
            except Unparseable as e:
                print("Невозможно распарсить HTML: {}".format(e))
                continue

    def paras(self, fileids=None, categories=None):
        """
        С использованием BeautifulSoup выделяет абзацы из HTML.

        Предполагается, что ведется работа с исходным не аннотированным корпусом.
        """

        for html in self.html(fileids, categories):
            soup = bs4.BeautifulSoup(html, 'lxml')  # указываем, что разбор происходить должен lxml-парсером
            for element in soup.find_all(TAGS):
                yield element.text
            soup.decompose()  # освобождаем память

    def sents(self, fileids=None, categories=None):
        """Выделяет предложения из абзацев с помощью NLTK функции sent_tokenize"""

        for paragraph in self.paras(fileids, categories):
            for sentence in sent_tokenize(paragraph):
                yield sentence

    def words(self, fileids=None, categories=None):
        """Выделяет слова из предложения с помощью NLTK функции wordpunct_tokenize"""

        for sentence in self.sents(fileids, categories):
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self, fileids=None, categories=None):
        """
        Сегментирует, лексемизирует и маркирует документ в корпусе с помощью NLTK фун-ии pos_tag

        Основное правило расшифроки тегов частей речи, которые возвращает pos_tag на АНГЛИЙСКОМ:
        N - если с N начинается тег, то это существительное
        V - глагол
        J - прилагательные
        R - наречия
        Все прочие теги определяют некоторые структурные элементы.

        Для РУССКОГО:
        S — существительное (яблоня, лошадь, корпус, вечность)
        A — прилагательное (коричневый, таинственный, морской)
        NUM — числительное (четыре, десять, много)
        ANUM — числительное-прилагательное (один, седьмой, восьмидесятый)
        V — глагол (пользоваться, обрабатывать)
        ADV — наречие (сгоряча, очень)
        PRAEDIC — предикатив (жаль, хорошо, пора)
        PARENTH — вводное слово (кстати, по-моему)
        SPRO — местоимение-существительное (она, что)
        APRO — местоимение-прилагательное (который, твой)
        ADVPRO — местоименное наречие (где, вот)
        PRAEDICPRO — местоимение-предикатив (некого, нечего)
        PR — предлог (под, напротив)
        CONJ — союз (и, чтобы)
        PART — частица (бы, же, пусть)
        INTJ — междометие (увы, батюшки)

        m — мужской род (работник, стол)
        f — женский род (работница, табуретка)
        m-f — «общий род» (задира, пьяница)
        n — средний род (животное, озеро)

        anim — одушевленность (человек, ангел, утопленник)
        inan — неодушевленность (рука, облако, культура)

        Значения грамматических категорий
        Род:
        m — мужской род (работник, стол)
        f — женский род (работница, табуретка)
        m-f — «общий род» (задира, пьяница)
        n — средний род (животное, озеро)

        Одушевленность:
        anim — одушевленность (человек, ангел, утопленник)
        inan — неодушевленность (рука, облако, культура)

        Число:
        sg — единственное число (яблоко, гордость)
        pl — множественное число (яблоки, ножницы, детишки)

        Падеж:
        nom — именительный падеж (голова, сын, степь, сани, который)
        gen — родительный падеж (головы, сына, степи, саней, которого)
        dat — дательный падеж (голове, сыну, степи, саням, которому)
        dat2 — дистрибутивный дательный ([по] многу, нескольку, стольку)
        acc — винительный падеж (голову, сына, степь, сани, который/которого)
        ins — творительный падеж (головой, сыном, степью, санями, которым)
        loc — предложный падеж ([о] голове, сыне, степи, санях, котором)
        gen2 — второй родительный падеж (чашка чаю)
        acc2 — второй винительный падеж (постричься в монахи; по два человека)
        loc2 — второй предложный падеж (в лесу, на оси́)
        voc — звательная форма (Господи, Серёж, ребят)
        adnum — счётная форма (два часа́, три шара́)

        Краткая/полная форма:
        brev — краткая форма (высок, нежна, прочны, рад)
        plen — полная форма (высокий, нежная, прочные, морской)
        Степень сравнения:
        comp — сравнительная степень (глубже)
        comp2 — форма «по+сравнительная степень» (поглубже)
        supr — превосходная степень (глубочайший)

        Вид:
        pf — совершенный вид (пошёл, встречу)
        ipf — несовершенный вид (ходил, встречаю)

        Переходность:
        intr — непереходность (ходить, вариться)
        tran — переходность (вести, варить)

        Залог:
        act — действительный залог (разрушил, разрушивший)
        pass — страдательный залог (только у причастий: разрушаемый, разрушенный)
        med — медиальный, или средний залог (глагольные формы на -ся: разрушился и т.п.)

        Форма (репрезентация) глагола:
        inf — инфинитив (украшать)
        partcp — причастие (украшенный)
        ger — деепричастие (украшая)

        Наклонение:
        indic — изъявительное наклонение (украшаю, украшал, украшу)
        imper — повелительное наклонение (украшай)
        imper2 — форма повелительного наклонения 1 л. мн. ч. на -те (идемте)

        Время:
        praet — прошедшее время (украшали, украшавший, украсив)
        praes — настоящее время (украшаем, украшающий, украшая)
        fut — будущее время (украсим)

        Лицо:
        1p — первое лицо (украшаю)
        2p — второе лицо (украшаешь)
        3p — третье лицо (украшает)

        Прочие признаки:
        persn — личное имя (Иван, Дарья, Леопольд, Эстер, Гомер, Маугли)
        patrn — отчество (Иванович, Павловна)
        famn — фамилия (Николаев, Волконская, Гумбольдт)
        zoon — кличка животного (Шарик, Дочка)
        0 — несклоняемое (шоссе, Седых)
        """

        for paragraph in self.paras(fileids, categories):
            yield [pos_tag(wordpunct_tokenize(sent), lang='rus') for sent in sent_tokenize(paragraph)]
