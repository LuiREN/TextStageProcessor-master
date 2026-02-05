#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
import os
import shutil
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

from sources.TextPreprocessing import loadInputFilesFromList, tokenizeTextData, removeStopWordsInTexts, \
    calculateWordsFrequencyInTexts, fixRegisterInTexts, normalizeTexts, writeStringToFile


def readConfigurationFile(filename):
    with codecs.open(filename, 'r', "utf-8") as text_file:
        data = text_file.read()
        lines = data.split("\n")
        result = dict()
        for line in lines:
            line = line.strip()
            if (line.startswith("#") == False):
                keyvalue = line.split("=")
                if (len(keyvalue) == 2):
                    result[keyvalue[0]] = keyvalue[1]
        return result


def getFilenameFromUserSelection(file_types="Any Files (*.*)", path=''):
    filenames, _ = QFileDialog.getOpenFileName(None, "Выбрать файл", path, file_types, None)
    if (len(filenames) > 0):
        return filenames
    else:
        return None


def getFilenamesFromUserSelection(path='', extensions: str = None):
    filenames, _ = QFileDialog.getOpenFileNames(None, "Выбрать файлы", path, "Text Files (*.txt);;{0}".format(
        extensions if not extensions == None else ''), None)
    if (len(filenames) > 0):
        return filenames
    else:
        return None


def getDirFromUserSelection(path):
    dir_name = QFileDialog.getExistingDirectory(None, "Выбрать каталог", path)
    if (len(dir_name) > 0):
        return dir_name
    else:
        return None


# Преобразует адреса вида /home/user/files/file.txt /home/user/files
# в вид: files/file.txt
def make_relative_files_path(filename, root_folder):
    folder = root_folder
    if (len(root_folder) > 0 and root_folder[-1] == '/'):
        folder = root_folder[:-1]
    start_position = folder.rfind('/')
    if (start_position == -1):
        start_position = 0
    if len(filename) > 0 and filename[0] == '/':
        start_position += 1
    return filename[start_position:]


def clear_dir(path):
    for name in os.listdir(path):
        full_name = path + name
        if (os.path.isfile(full_name)):
            os.remove(full_name)
        else:
            shutil.rmtree(full_name)


def makePreprocessingForAllFilesInFolder(configurations,
                                         input_dir_name,
                                         output_files_dir,
                                         output_log_dir, morph):
    input_filenames_with_dir = []

    not_a_folder = input_dir_name + '.DS_Store'
    if os.path.exists(not_a_folder):
        os.remove(not_a_folder)

    for top, dirs, files in os.walk(input_dir_name):
        for nm in files:
            if nm != '.DS_Store':
                input_filenames_with_dir.append(os.path.join(top, nm))

    # Загружаем предложения из нескольких файлов
    texts = loadInputFilesFromList(input_filenames_with_dir)

    for text in texts:
        text.short_filename = make_relative_files_path(text.full_filename, input_dir_name)

    # Разделяем предложения на слова
    texts = tokenizeTextData(texts)

    print('Этап препроцессинга:')

    print('1) Удаление стоп-слов.')
    texts, log_string = removeStopWordsInTexts(texts, morph, configurations)
    writeStringToFile(log_string.replace('\n ', '\n'), output_log_dir + '/output_stage_1.txt')

    # Переводим обычное предложение в нормализованное (каждое слово)
    print('2) Нормализация.')
    texts, log_string = normalizeTexts(texts, morph)
    writeStringToFile(log_string.replace('\n ', '\n'), output_log_dir + '/output_stage_2.txt')

    # Приведение регистра (все слова с маленькой буквы за исключением ФИО)
    print('3) Приведение регистра.')
    texts, log_string = fixRegisterInTexts(texts, morph)
    writeStringToFile(log_string.replace('\n ', '\n'), output_log_dir + '/output_stage_3.txt')

    # Подсчет частоты слов в тексте
    print('4) Расчет частотной таблицы слов.')
    texts, log_string = calculateWordsFrequencyInTexts(texts)
    writeStringToFile(log_string.replace('\n ', '\n'), output_log_dir + '/output_stage_4.csv')

    for text in texts:
        text_filename = output_files_dir + text.short_filename
        os.makedirs(os.path.dirname(text_filename), exist_ok=True)
        with open(text_filename, 'w', encoding='utf-8') as out_text_file:
            for sentence in text.register_pass_centences:
                for word in sentence:
                    out_text_file.write(word)
                    out_text_file.write(' ')


# Измерение времени выполнения блока кода
class Profiler(object):

    def __init__(self):
        self._startTime = 0

    def start(self):
        self._startTime = time.time()

    def stop(self):
        return str("{:.3f}").format(time.time() - self._startTime)


# === Дополнительные утилиты для модулей-диалогов ===

import re
import csv
import math
import sys
import numpy as np
from collections import Counter

try:
    import pymorphy2
except (ImportError, AttributeError):
    import pymorphy3 as pymorphy2


def readTextFile(filename):
    """Чтение текстового файла с автоопределением кодировки."""
    encodings = ['utf-8', 'cp1251', 'latin-1']
    for enc in encodings:
        try:
            with open(filename, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def readStopWords(filename):
    """Чтение файла стоп-слов."""
    stop_words = set()
    try:
        text = readTextFile(filename)
        for line in text.split('\n'):
            word = line.strip().lower()
            if word:
                stop_words.add(word)
    except FileNotFoundError:
        print(f"Файл стоп-слов {filename} не найден.")
    return stop_words


def tokenize(text):
    """Токенизация текста на слова."""
    return re.findall(r'[а-яА-ЯёЁa-zA-Z]+', text.lower())


def lemmatize(tokens, morph):
    """Лемматизация списка токенов."""
    lemmas = []
    for token in tokens:
        parsed = morph.parse(token)
        if parsed:
            lemmas.append(parsed[0].normal_form)
        else:
            lemmas.append(token)
    return lemmas


def removeStopWords(tokens, stop_words):
    """Удаление стоп-слов из списка токенов."""
    return [t for t in tokens if t.lower() not in stop_words]


def preprocessText(text, morph, stop_words):
    """Полная предобработка текста: токенизация, лемматизация, удаление стоп-слов."""
    tokens = tokenize(text)
    tokens = removeStopWords(tokens, stop_words)
    lemmas = lemmatize(tokens, morph)
    lemmas = removeStopWords(lemmas, stop_words)
    return lemmas


def buildTFIDF(documents):
    """Построение TF-IDF матрицы для списка документов (списков слов)."""
    doc_count = len(documents)
    all_words = sorted(set(word for doc in documents for word in doc))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    tf_matrix = np.zeros((doc_count, len(all_words)))
    for i, doc in enumerate(documents):
        counter = Counter(doc)
        total = len(doc) if len(doc) > 0 else 1
        for word, count in counter.items():
            if word in word_to_idx:
                tf_matrix[i][word_to_idx[word]] = count / total

    idf = np.zeros(len(all_words))
    for j, word in enumerate(all_words):
        doc_freq = sum(1 for doc in documents if word in doc)
        idf[j] = math.log(doc_count / (1 + doc_freq)) + 1

    tfidf_matrix = tf_matrix * idf
    return tfidf_matrix, all_words


def cosineSimilarity(vec1, vec2):
    """Вычисление косинусного сходства между двумя векторами."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def writeResultToCSV(filename, headers, rows):
    """Запись результатов в CSV файл."""
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def sentenceTokenize(text):
    """Разбиение текста на предложения."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]
