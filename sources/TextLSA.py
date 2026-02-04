#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль латентно-семантического анализа (LSA)."""

import os
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sources.utils import (
    readTextFile, readStopWords, preprocessText, writeResultToCSV
)


class LSAThread(QThread):
    """Поток для выполнения LSA."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, filenames, morph, configurations, n_components):
        super().__init__()
        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.n_components = n_components

    def run(self):
        try:
            stop_words_file = self.configurations.get("stop_words_filename", "sources/russian_stop_words.txt")
            stop_words = readStopWords(stop_words_file)

            self.log_signal.emit("Чтение и предобработка документов...")
            doc_texts = []
            doc_names = []

            for i, filename in enumerate(self.filenames):
                text = readTextFile(filename)
                tokens = preprocessText(text, self.morph, stop_words)
                doc_texts.append(' '.join(tokens))
                doc_names.append(os.path.basename(filename))
                self.progress.emit(int((i + 1) / len(self.filenames) * 30))

            self.log_signal.emit("Построение TF-IDF матрицы...")
            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(doc_texts)
            feature_names = vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else []
            self.progress.emit(50)

            n_components = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
            if n_components < 1:
                n_components = 1

            self.log_signal.emit(f"Выполнение SVD (компонент: {n_components})...")
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            doc_topics = svd.fit_transform(tfidf_matrix)
            self.progress.emit(70)

            explained_variance = svd.explained_variance_ratio_
            components = svd.components_

            # Извлечение ключевых слов для каждой темы
            topics_keywords = []
            for i, component in enumerate(components):
                top_indices = component.argsort()[-10:][::-1]
                top_words = [(feature_names[idx], component[idx]) for idx in top_indices]
                topics_keywords.append(top_words)

            # Матрица сходства документов в LSA пространстве
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(doc_topics)

            self.progress.emit(90)

            result = {
                'doc_names': doc_names,
                'doc_topics': doc_topics,
                'topics_keywords': topics_keywords,
                'explained_variance': explained_variance,
                'similarity_matrix': similarity_matrix,
                'n_components': n_components
            }

            self.progress.emit(100)
            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(str(e))


class DialogConfigLSA(QDialog):
    """Диалог настройки и выполнения LSA."""

    def __init__(self, filenames, morph, configurations, parent=None):
        super().__init__(parent)
        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Латентно-семантический анализ (LSA)")
        self.setMinimumSize(700, 600)

        layout = QVBoxLayout()

        settings_group = QGroupBox("Параметры LSA")
        form_layout = QFormLayout()

        self.spin_components = QSpinBox()
        self.spin_components.setRange(1, 100)
        self.spin_components.setValue(5)
        form_layout.addRow("Количество тем (компонент):", self.spin_components)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        info_label = QLabel(f"Выбрано файлов: {len(self.filenames)}")
        layout.addWidget(info_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Начать анализ")
        self.btn_start.clicked.connect(self.startLSA)
        btn_layout.addWidget(self.btn_start)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        layout.addWidget(self.text_result)

        self.setLayout(layout)

    def startLSA(self):
        self.btn_start.setEnabled(False)
        self.text_result.clear()
        self.progress_bar.setValue(0)

        self.thread = LSAThread(
            self.filenames, self.morph, self.configurations,
            self.spin_components.value()
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(lambda t: self.text_result.append(t))
        self.thread.finished_signal.connect(self.onFinished)
        self.thread.error_signal.connect(self.onError)
        self.thread.start()

    def onFinished(self, result):
        self.progress_bar.setValue(100)
        self.btn_start.setEnabled(True)

        self.text_result.append(f"\n=== Результаты LSA ({result['n_components']} тем) ===\n")

        # Объяснённая дисперсия
        total_var = sum(result['explained_variance'])
        self.text_result.append(f"Общая объяснённая дисперсия: {total_var:.4f}\n")

        # Темы и ключевые слова
        for i, keywords in enumerate(result['topics_keywords']):
            var = result['explained_variance'][i]
            self.text_result.append(f"Тема {i + 1} (дисперсия: {var:.4f}):")
            words = [f"  {word} ({weight:.4f})" for word, weight in keywords]
            self.text_result.append('\n'.join(words))
            self.text_result.append("")

        # Матрица сходства документов
        self.text_result.append("Матрица сходства документов (косинусное):")
        sim = result['similarity_matrix']
        names = result['doc_names']
        header = "          " + "  ".join([n[:8].ljust(8) for n in names])
        self.text_result.append(header)
        for i, name in enumerate(names):
            row = name[:8].ljust(10)
            row += "  ".join([f"{sim[i][j]:.4f}  " for j in range(len(names))])
            self.text_result.append(row)

        # Сохранение
        output_dir = self.configurations.get("output_files_directory", "output_files")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "lsa_result.csv")

        headers = ["Документ"] + [f"Тема_{i + 1}" for i in range(result['n_components'])]
        rows = []
        for i, name in enumerate(result['doc_names']):
            row = [name] + [f"{v:.4f}" for v in result['doc_topics'][i]]
            rows.append(row)
        writeResultToCSV(output_file, headers, rows)
        self.text_result.append(f"\nРезультаты сохранены в: {output_file}")

    def onError(self, error_text):
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_text}")
