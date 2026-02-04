#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль классификации текстов (собственная реализация)."""

import os
import numpy as np
from collections import Counter

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from sources.utils import (
    readTextFile, readStopWords, preprocessText,
    buildTFIDF, cosineSimilarity, writeResultToCSV
)


class ClassificationThread(QThread):
    """Поток для выполнения классификации."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, dirname, morph, configurations, method, k_neighbors):
        super().__init__()
        self.dirname = dirname
        self.morph = morph
        self.configurations = configurations
        self.method = method
        self.k_neighbors = k_neighbors

    def run(self):
        try:
            stop_words_file = self.configurations.get("stop_words_filename", "sources/russian_stop_words.txt")
            stop_words = readStopWords(stop_words_file)

            self.log_signal.emit("Чтение обучающих данных...")

            # Читаем данные из поддиректорий (каждая = класс)
            classes = []
            documents = []
            doc_names = []
            doc_classes = []

            subdirs = [d for d in os.listdir(self.dirname)
                       if os.path.isdir(os.path.join(self.dirname, d))]

            if len(subdirs) < 2:
                self.error_signal.emit("Необходимо минимум 2 класса (поддиректории)")
                return

            for subdir in sorted(subdirs):
                class_name = subdir
                classes.append(class_name)
                class_dir = os.path.join(self.dirname, subdir)
                files = [f for f in os.listdir(class_dir) if f.endswith('.txt')]

                for filename in files:
                    filepath = os.path.join(class_dir, filename)
                    text = readTextFile(filepath)
                    tokens = preprocessText(text, self.morph, stop_words)
                    documents.append(tokens)
                    doc_names.append(filename)
                    doc_classes.append(class_name)

            self.progress.emit(30)
            self.log_signal.emit(f"Загружено {len(documents)} документов, {len(classes)} классов")
            self.log_signal.emit("Построение TF-IDF матрицы...")

            tfidf_matrix, feature_names = buildTFIDF(documents)
            self.progress.emit(50)

            self.log_signal.emit("Классификация методом KNN (Leave-One-Out)...")

            # KNN с Leave-One-Out
            correct = 0
            predictions = []
            n = len(documents)

            for i in range(n):
                similarities = []
                for j in range(n):
                    if i != j:
                        sim = cosineSimilarity(tfidf_matrix[i], tfidf_matrix[j])
                        similarities.append((sim, doc_classes[j]))

                similarities.sort(key=lambda x: -x[0])
                top_k = similarities[:self.k_neighbors]

                class_votes = Counter([c for _, c in top_k])
                predicted = class_votes.most_common(1)[0][0]
                predictions.append(predicted)

                if predicted == doc_classes[i]:
                    correct += 1

                self.progress.emit(50 + int((i + 1) / n * 40))

            accuracy = correct / n if n > 0 else 0

            result = {
                'accuracy': accuracy,
                'correct': correct,
                'total': n,
                'predictions': predictions,
                'actual': doc_classes,
                'doc_names': doc_names,
                'classes': classes,
                'method': self.method
            }

            self.progress.emit(100)
            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(str(e))


class DialogConfigClassification(QDialog):
    """Диалог настройки и выполнения классификации."""

    def __init__(self, dirname, morph, configurations, parent=None):
        super().__init__(parent)
        self.dirname = dirname
        self.morph = morph
        self.configurations = configurations
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Классификация текстов")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout()

        settings_group = QGroupBox("Параметры классификации")
        form_layout = QFormLayout()

        self.combo_method = QComboBox()
        self.combo_method.addItems(["KNN (K ближайших соседей)"])
        form_layout.addRow("Метод:", self.combo_method)

        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 50)
        self.spin_k.setValue(5)
        form_layout.addRow("K (количество соседей):", self.spin_k)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        info_label = QLabel(f"Директория: {self.dirname}")
        layout.addWidget(info_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Начать классификацию")
        self.btn_start.clicked.connect(self.startClassification)
        btn_layout.addWidget(self.btn_start)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        layout.addWidget(self.text_result)

        self.setLayout(layout)

    def startClassification(self):
        self.btn_start.setEnabled(False)
        self.text_result.clear()
        self.progress_bar.setValue(0)

        self.thread = ClassificationThread(
            self.dirname, self.morph, self.configurations,
            self.combo_method.currentText(),
            self.spin_k.value()
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(lambda t: self.text_result.append(t))
        self.thread.finished_signal.connect(self.onFinished)
        self.thread.error_signal.connect(self.onError)
        self.thread.start()

    def onFinished(self, result):
        self.progress_bar.setValue(100)
        self.btn_start.setEnabled(True)

        self.text_result.append(f"\n=== Результаты классификации ===\n")
        self.text_result.append(f"Точность: {result['accuracy']:.2%} ({result['correct']}/{result['total']})\n")

        self.text_result.append("Детальные результаты:")
        for i, name in enumerate(result['doc_names']):
            actual = result['actual'][i]
            predicted = result['predictions'][i]
            mark = "+" if actual == predicted else "ОШИБКА"
            self.text_result.append(f"  {name}: факт={actual}, прогноз={predicted} [{mark}]")

        output_dir = self.configurations.get("output_files_directory", "output_files")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "classification_result.csv")

        headers = ["Документ", "Фактический класс", "Предсказанный класс", "Верно"]
        rows = []
        for i, name in enumerate(result['doc_names']):
            rows.append([name, result['actual'][i], result['predictions'][i],
                         "Да" if result['actual'][i] == result['predictions'][i] else "Нет"])
        writeResultToCSV(output_file, headers, rows)
        self.text_result.append(f"\nРезультаты сохранены в: {output_file}")

    def onError(self, error_text):
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_text}")
