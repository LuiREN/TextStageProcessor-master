#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль кластеризации текстов с использованием библиотек scikit-learn."""

import os
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from sources.utils import (
    readTextFile, readStopWords, preprocessText, writeResultToCSV
)


class ClasteringThread(QThread):
    """Поток для выполнения кластеризации с использованием sklearn."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, filenames, morph, configurations, n_clusters, algorithm):
        super().__init__()
        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.n_clusters = n_clusters
        self.algorithm = algorithm

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

            self.log_signal.emit("Векторизация TF-IDF (sklearn)...")
            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(doc_texts)
            self.progress.emit(50)

            self.log_signal.emit(f"Кластеризация методом {self.algorithm}...")

            if self.algorithm == "K-Means":
                model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(tfidf_matrix)
            elif self.algorithm == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=self.n_clusters)
                labels = model.fit_predict(tfidf_matrix.toarray())
            elif self.algorithm == "DBSCAN":
                model = DBSCAN(eps=0.5, min_samples=2)
                labels = model.fit_predict(tfidf_matrix.toarray())
            else:
                model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(tfidf_matrix)

            self.progress.emit(80)

            # Оценка качества
            silhouette = -1
            n_labels = len(set(labels)) - (1 if -1 in labels else 0)
            if 1 < n_labels < len(doc_names):
                silhouette = silhouette_score(tfidf_matrix, labels)

            clusters = {}
            for i, label in enumerate(labels):
                label_key = int(label)
                if label_key not in clusters:
                    clusters[label_key] = []
                clusters[label_key].append(doc_names[i])

            result = {
                'labels': labels,
                'doc_names': doc_names,
                'clusters': clusters,
                'silhouette': silhouette,
                'algorithm': self.algorithm,
                'feature_names': vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else []
            }

            self.progress.emit(100)
            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(str(e))


class DialogClastering(QDialog):
    """Диалог кластеризации с использованием sklearn."""

    def __init__(self, filenames, morph, configurations, parent=None):
        super().__init__(parent)
        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Кластеризация (LIB)")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout()

        settings_group = QGroupBox("Параметры кластеризации")
        form_layout = QFormLayout()

        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems(["K-Means", "Agglomerative", "DBSCAN"])
        form_layout.addRow("Алгоритм:", self.combo_algorithm)

        self.spin_clusters = QSpinBox()
        self.spin_clusters.setRange(2, 100)
        self.spin_clusters.setValue(3)
        form_layout.addRow("Количество кластеров:", self.spin_clusters)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        info_label = QLabel(f"Выбрано файлов: {len(self.filenames)}")
        layout.addWidget(info_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Начать кластеризацию")
        self.btn_start.clicked.connect(self.startClastering)
        btn_layout.addWidget(self.btn_start)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        layout.addWidget(self.text_result)

        self.setLayout(layout)

    def startClastering(self):
        self.btn_start.setEnabled(False)
        self.text_result.clear()
        self.progress_bar.setValue(0)

        self.thread = ClasteringThread(
            self.filenames, self.morph, self.configurations,
            self.spin_clusters.value(),
            self.combo_algorithm.currentText()
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(lambda t: self.text_result.append(t))
        self.thread.finished_signal.connect(self.onFinished)
        self.thread.error_signal.connect(self.onError)
        self.thread.start()

    def onFinished(self, result):
        self.progress_bar.setValue(100)
        self.btn_start.setEnabled(True)

        self.text_result.append(f"\n=== Результаты кластеризации ({result['algorithm']}) ===\n")

        if result['silhouette'] >= 0:
            self.text_result.append(f"Silhouette Score: {result['silhouette']:.4f}\n")

        clusters = result['clusters']
        for cluster_id in sorted(clusters.keys()):
            docs = clusters[cluster_id]
            label = f"Кластер {cluster_id + 1}" if cluster_id >= 0 else "Шум (не кластеризовано)"
            self.text_result.append(f"{label} ({len(docs)} документов):")
            for doc in docs:
                self.text_result.append(f"  - {doc}")
            self.text_result.append("")

        output_dir = self.configurations.get("output_files_directory", "output_files")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "clastering_lib_result.csv")

        headers = ["Документ", "Кластер"]
        rows = [[name, int(label) + 1] for name, label in zip(result['doc_names'], result['labels'])]
        writeResultToCSV(output_file, headers, rows)
        self.text_result.append(f"Результаты сохранены в: {output_file}")

    def onError(self, error_text):
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_text}")
