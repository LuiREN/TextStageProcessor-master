#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import re
import math
import codecs
from collections import Counter

try:
    import pymorphy2
except (ImportError, AttributeError):
    import pymorphy3 as pymorphy2
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QDialog,
    QLabel, QLineEdit, QPushButton, QTextEdit, QPlainTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton,
    QGroupBox, QFileDialog, QMessageBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QSpacerItem,
    QGridLayout, QFormLayout, QTabWidget, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import QtCore


def readConfigurationFile(filename):
    """Чтение конфигурационного файла в словарь."""
    configurations = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        configurations[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Конфигурационный файл {filename} не найден. Используются значения по умолчанию.")
    return configurations


def getFilenamesFromUserSelection(start_dir=None):
    """Открывает диалог выбора нескольких файлов."""
    if start_dir and not os.path.exists(start_dir):
        os.makedirs(start_dir)
    filenames, _ = QFileDialog.getOpenFileNames(
        None,
        "Выберите файлы",
        start_dir if start_dir else "",
        "Text files (*.txt);;All files (*.*)"
    )
    if filenames:
        return filenames
    return None


def getFilenameFromUserSelection(file_filter="Text file (*.txt)", start_dir=None):
    """Открывает диалог выбора одного файла."""
    if start_dir and not os.path.exists(start_dir):
        os.makedirs(start_dir)
    filename, _ = QFileDialog.getOpenFileName(
        None,
        "Выберите файл",
        start_dir if start_dir else "",
        file_filter
    )
    if filename:
        return filename
    return None


def getDirFromUserSelection(start_dir=None):
    """Открывает диалог выбора директории."""
    if start_dir and not os.path.exists(start_dir):
        os.makedirs(start_dir)
    dirname = QFileDialog.getExistingDirectory(
        None,
        "Выберите директорию",
        start_dir if start_dir else ""
    )
    if dirname:
        return dirname
    return None


def readTextFile(filename):
    """Чтение текстового файла."""
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

    # TF
    tf_matrix = np.zeros((doc_count, len(all_words)))
    for i, doc in enumerate(documents):
        counter = Counter(doc)
        total = len(doc) if len(doc) > 0 else 1
        for word, count in counter.items():
            if word in word_to_idx:
                tf_matrix[i][word_to_idx[word]] = count / total

    # IDF
    idf = np.zeros(len(all_words))
    for j, word in enumerate(all_words):
        doc_freq = sum(1 for doc in documents if word in doc)
        idf[j] = math.log(doc_count / (1 + doc_freq)) + 1

    # TF-IDF
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
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def sentenceTokenize(text):
    """Разбиение текста на предложения."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


class WorkerThread(QThread):
    """Базовый рабочий поток для фоновых вычислений."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))
