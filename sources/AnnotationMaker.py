#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль создания аннотации (реферата) документа."""

import os
import re
import numpy as np
from collections import Counter

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from sources.utils import (
    readTextFile, readStopWords, preprocessText,
    tokenize, lemmatize, sentenceTokenize, writeResultToCSV,
    cosineSimilarity
)


class AnnotationThread(QThread):
    """Поток для создания аннотации."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, filename, morph, configurations, num_sentences, method):
        super().__init__()
        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.num_sentences = num_sentences
        self.method = method

    def run(self):
        try:
            stop_words_file = self.configurations.get("stop_words_filename", "sources/russian_stop_words.txt")
            stop_words = readStopWords(stop_words_file)

            self.log_signal.emit("Чтение документа...")
            text = readTextFile(self.filename)
            sentences = sentenceTokenize(text)

            if not sentences:
                self.error_signal.emit("Текст не содержит предложений")
                return

            self.progress.emit(20)
            self.log_signal.emit(f"Найдено предложений: {len(sentences)}")

            num_sentences = min(self.num_sentences, len(sentences))

            if self.method == "Частотный":
                selected_indices = self._frequencyMethod(sentences, stop_words, num_sentences)
            elif self.method == "Позиционный":
                selected_indices = self._positionMethod(sentences, num_sentences)
            else:
                selected_indices = self._combinedMethod(sentences, stop_words, num_sentences)

            self.progress.emit(80)

            # Сортируем по порядку в тексте
            selected_indices.sort()
            annotation_sentences = [sentences[i] for i in selected_indices]
            annotation = '. '.join(annotation_sentences) + '.'

            result = {
                'original_text': text,
                'original_sentences': len(sentences),
                'annotation': annotation,
                'selected_sentences': annotation_sentences,
                'selected_indices': selected_indices,
                'compression': len(annotation) / len(text) if len(text) > 0 else 0,
                'method': self.method
            }

            self.progress.emit(100)
            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(str(e))

    def _frequencyMethod(self, sentences, stop_words, num_sentences):
        """Частотный метод: выбираем предложения с наиболее частотными словами."""
        # Подсчёт частот слов во всём тексте
        all_tokens = []
        for sent in sentences:
            tokens = preprocessText(sent, self.morph, stop_words)
            all_tokens.extend(tokens)

        word_freq = Counter(all_tokens)

        # Оценка каждого предложения
        scores = []
        for i, sent in enumerate(sentences):
            tokens = preprocessText(sent, self.morph, stop_words)
            if not tokens:
                scores.append(0)
                continue
            score = sum(word_freq.get(t, 0) for t in tokens) / len(tokens)
            scores.append(score)
            self.progress.emit(20 + int((i + 1) / len(sentences) * 50))

        top_indices = np.argsort(scores)[-num_sentences:]
        return list(top_indices)

    def _positionMethod(self, sentences, num_sentences):
        """Позиционный метод: выбираем предложения из начала, середины и конца."""
        n = len(sentences)
        indices = set()

        # Первые предложения
        for i in range(min(num_sentences // 3 + 1, n)):
            indices.add(i)

        # Средние предложения
        mid = n // 2
        for i in range(max(0, mid - num_sentences // 6), min(n, mid + num_sentences // 6 + 1)):
            if len(indices) < num_sentences:
                indices.add(i)

        # Последние предложения
        for i in range(max(0, n - num_sentences // 3), n):
            if len(indices) < num_sentences:
                indices.add(i)

        # Добрать если нужно
        idx = 0
        while len(indices) < num_sentences and idx < n:
            indices.add(idx)
            idx += 1

        return list(indices)[:num_sentences]

    def _combinedMethod(self, sentences, stop_words, num_sentences):
        """Комбинированный метод: частотный + позиционный."""
        all_tokens = []
        for sent in sentences:
            tokens = preprocessText(sent, self.morph, stop_words)
            all_tokens.extend(tokens)

        word_freq = Counter(all_tokens)
        n = len(sentences)

        scores = []
        for i, sent in enumerate(sentences):
            tokens = preprocessText(sent, self.morph, stop_words)
            if not tokens:
                scores.append(0)
                continue

            # Частотная оценка
            freq_score = sum(word_freq.get(t, 0) for t in tokens) / len(tokens)

            # Позиционная оценка (начало и конец важнее)
            position = i / n
            if position < 0.2:
                pos_score = 1.0
            elif position > 0.8:
                pos_score = 0.8
            else:
                pos_score = 0.5

            # Оценка длины (средние предложения предпочтительнее)
            len_score = min(len(tokens) / 20.0, 1.0)

            combined = freq_score * 0.5 + pos_score * 0.3 + len_score * 0.2
            scores.append(combined)
            self.progress.emit(20 + int((i + 1) / len(sentences) * 50))

        top_indices = np.argsort(scores)[-num_sentences:]
        return list(top_indices)


class DialogAnnotationMaker(QDialog):
    """Диалог создания аннотации документа."""

    def __init__(self, filename, morph, configurations, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Создание аннотации документа")
        self.setMinimumSize(700, 600)

        layout = QVBoxLayout()

        settings_group = QGroupBox("Параметры аннотирования")
        form_layout = QFormLayout()

        self.spin_sentences = QSpinBox()
        self.spin_sentences.setRange(1, 50)
        self.spin_sentences.setValue(5)
        form_layout.addRow("Количество предложений:", self.spin_sentences)

        self.combo_method = QComboBox()
        self.combo_method.addItems(["Комбинированный", "Частотный", "Позиционный"])
        form_layout.addRow("Метод аннотирования:", self.combo_method)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        info_label = QLabel(f"Файл: {os.path.basename(self.filename)}")
        layout.addWidget(info_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Создать аннотацию")
        self.btn_start.clicked.connect(self.startAnnotation)
        btn_layout.addWidget(self.btn_start)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        layout.addWidget(self.text_result)

        self.setLayout(layout)

    def startAnnotation(self):
        self.btn_start.setEnabled(False)
        self.text_result.clear()
        self.progress_bar.setValue(0)

        self.thread = AnnotationThread(
            self.filename, self.morph, self.configurations,
            self.spin_sentences.value(),
            self.combo_method.currentText()
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(lambda t: self.text_result.append(t))
        self.thread.finished_signal.connect(self.onFinished)
        self.thread.error_signal.connect(self.onError)
        self.thread.start()

    def onFinished(self, result):
        self.progress_bar.setValue(100)
        self.btn_start.setEnabled(True)

        self.text_result.append(f"=== Аннотация документа (метод: {result['method']}) ===\n")
        self.text_result.append(f"Исходный текст: {result['original_sentences']} предложений")
        self.text_result.append(f"Аннотация: {len(result['selected_sentences'])} предложений")
        self.text_result.append(f"Степень сжатия: {result['compression']:.2%}\n")
        self.text_result.append("--- Аннотация ---\n")
        self.text_result.append(result['annotation'])

        output_dir = self.configurations.get("output_files_directory", "output_files")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "annotation_result.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['annotation'])
        self.text_result.append(f"\nАннотация сохранена в: {output_file}")

    def onError(self, error_text):
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_text}")
