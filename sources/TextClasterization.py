#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль кластеризации текстов (собственная реализация K-means)."""

import os
import numpy as np
from collections import Counter

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from sources.utils import (
    readTextFile, readStopWords, preprocessText,
    buildTFIDF, cosineSimilarity, writeResultToCSV
)


class ClasterizationThread(QThread):
    """Поток для выполнения кластеризации."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, filenames, morph, configurations, n_clusters, max_iter, distance_metric):
        super().__init__()
        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance_metric = distance_metric

    def run(self):
        try:
            stop_words_file = self.configurations.get("stop_words_filename", "sources/russian_stop_words.txt")
            stop_words = readStopWords(stop_words_file)

            self.log_signal.emit("Чтение и предобработка документов...")
            documents = []
            doc_names = []
            for i, filename in enumerate(self.filenames):
                text = readTextFile(filename)
                tokens = preprocessText(text, self.morph, stop_words)
                documents.append(tokens)
                doc_names.append(os.path.basename(filename))
                self.progress.emit(int((i + 1) / len(self.filenames) * 30))

            self.log_signal.emit("Построение TF-IDF матрицы...")
            tfidf_matrix, feature_names = buildTFIDF(documents)
            self.progress.emit(50)

            self.log_signal.emit(f"Кластеризация K-means (k={self.n_clusters})...")
            labels = self._kmeans(tfidf_matrix, self.n_clusters, self.max_iter)
            self.progress.emit(90)

            # Формируем результат
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(doc_names[i])

            result = {
                'labels': labels,
                'doc_names': doc_names,
                'clusters': clusters,
                'n_clusters': self.n_clusters,
                'tfidf_matrix': tfidf_matrix,
                'feature_names': feature_names
            }

            self.progress.emit(100)
            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(str(e))

    def _kmeans(self, data, k, max_iter=100):
        """Простая реализация K-means."""
        n_samples = data.shape[0]
        if n_samples < k:
            k = n_samples

        # Инициализация центроидов случайным выбором
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = data[indices].copy()

        labels = np.zeros(n_samples, dtype=int)

        for iteration in range(max_iter):
            # Присваиваем каждый документ ближайшему центроиду
            new_labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                min_dist = float('inf')
                for j in range(k):
                    if self.distance_metric == "euclidean":
                        dist = np.linalg.norm(data[i] - centroids[j])
                    else:
                        dist = 1 - cosineSimilarity(data[i], centroids[j])
                    if dist < min_dist:
                        min_dist = dist
                        new_labels[i] = j

            # Проверяем сходимость
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Пересчитываем центроиды
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    centroids[j] = data[mask].mean(axis=0)

            progress_val = 50 + int((iteration + 1) / max_iter * 40)
            self.progress.emit(min(progress_val, 89))

        return labels


class DialogConfigClasterization(QDialog):
    """Диалог настройки и выполнения кластеризации."""

    def __init__(self, filenames, morph, configurations, parent=None):
        super().__init__(parent)
        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Кластеризация текстов")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout()

        # Настройки
        settings_group = QGroupBox("Параметры кластеризации")
        form_layout = QFormLayout()

        self.spin_clusters = QSpinBox()
        self.spin_clusters.setRange(2, 100)
        self.spin_clusters.setValue(3)
        form_layout.addRow("Количество кластеров:", self.spin_clusters)

        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(10, 1000)
        self.spin_iterations.setValue(100)
        form_layout.addRow("Макс. итераций:", self.spin_iterations)

        self.combo_distance = QComboBox()
        self.combo_distance.addItems(["cosine", "euclidean"])
        form_layout.addRow("Метрика расстояния:", self.combo_distance)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        info_label = QLabel(f"Выбрано файлов: {len(self.filenames)}")
        layout.addWidget(info_label)

        # Прогресс
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Кнопки
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Начать кластеризацию")
        self.btn_start.clicked.connect(self.startClasterization)
        btn_layout.addWidget(self.btn_start)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        # Результаты
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        layout.addWidget(self.text_result)

        self.setLayout(layout)

    def startClasterization(self):
        self.btn_start.setEnabled(False)
        self.text_result.clear()
        self.progress_bar.setValue(0)

        n_clusters = self.spin_clusters.value()
        max_iter = self.spin_iterations.value()
        distance = self.combo_distance.currentText()

        self.thread = ClasterizationThread(
            self.filenames, self.morph, self.configurations,
            n_clusters, max_iter, distance
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(self.appendLog)
        self.thread.finished_signal.connect(self.onFinished)
        self.thread.error_signal.connect(self.onError)
        self.thread.start()

    def appendLog(self, text):
        self.text_result.append(text)

    def onFinished(self, result):
        self.progress_bar.setValue(100)
        self.btn_start.setEnabled(True)

        self.text_result.append("\n=== Результаты кластеризации ===\n")
        clusters = result['clusters']
        for cluster_id in sorted(clusters.keys()):
            docs = clusters[cluster_id]
            self.text_result.append(f"Кластер {cluster_id + 1} ({len(docs)} документов):")
            for doc in docs:
                self.text_result.append(f"  - {doc}")
            self.text_result.append("")

        # Сохранение результатов
        output_dir = self.configurations.get("output_files_directory", "output_files")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "clasterization_result.csv")

        headers = ["Документ", "Кластер"]
        rows = []
        for i, name in enumerate(result['doc_names']):
            rows.append([name, result['labels'][i] + 1])
        writeResultToCSV(output_file, headers, rows)
        self.text_result.append(f"Результаты сохранены в: {output_file}")

    def onError(self, error_text):
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_text}")
