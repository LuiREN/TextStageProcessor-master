#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль классификации текстов с использованием библиотек scikit-learn."""

import os
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import classification_report, accuracy_score

from sources.utils import (
    readTextFile, readStopWords, preprocessText, writeResultToCSV
)


class ClassificationLibThread(QThread):
    """Поток для классификации с использованием sklearn."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, dirname, morph, configurations, algorithm, cv_folds):
        super().__init__()
        self.dirname = dirname
        self.morph = morph
        self.configurations = configurations
        self.algorithm = algorithm
        self.cv_folds = cv_folds

    def run(self):
        try:
            stop_words_file = self.configurations.get("stop_words_filename", "sources/russian_stop_words.txt")
            stop_words = readStopWords(stop_words_file)

            self.log_signal.emit("Чтение обучающих данных...")

            doc_texts = []
            doc_labels = []
            doc_names = []
            classes = []

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
                    doc_texts.append(' '.join(tokens))
                    doc_labels.append(class_name)
                    doc_names.append(filename)

            self.progress.emit(30)
            self.log_signal.emit(f"Загружено {len(doc_texts)} документов, {len(classes)} классов")
            self.log_signal.emit("Векторизация TF-IDF...")

            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(doc_texts)
            y = np.array(doc_labels)
            self.progress.emit(50)

            self.log_signal.emit(f"Классификация методом {self.algorithm}...")

            if self.algorithm == "Naive Bayes":
                model = MultinomialNB()
            elif self.algorithm == "SVM":
                model = SVC(kernel='linear', random_state=42)
            elif self.algorithm == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif self.algorithm == "KNN":
                model = KNeighborsClassifier(n_neighbors=5)
            else:
                model = MultinomialNB()

            self.progress.emit(60)

            # Кросс-валидация
            n_folds = min(self.cv_folds, len(doc_texts))
            if n_folds < 2:
                n_folds = 2

            scores = cross_val_score(model, X, y, cv=n_folds)
            self.progress.emit(80)

            # Обучение на всех данных и предсказание
            model.fit(X, y)
            predictions = model.predict(X)

            report = classification_report(y, predictions, output_dict=True)
            report_text = classification_report(y, predictions)

            result = {
                'cv_scores': scores,
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'predictions': predictions.tolist(),
                'actual': doc_labels,
                'doc_names': doc_names,
                'classes': classes,
                'algorithm': self.algorithm,
                'report': report_text,
                'accuracy': accuracy_score(y, predictions)
            }

            self.progress.emit(100)
            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(str(e))


class DialogClassificationLib(QDialog):
    """Диалог классификации с использованием sklearn."""

    def __init__(self, dirname, morph, configurations, parent=None):
        super().__init__(parent)
        self.dirname = dirname
        self.morph = morph
        self.configurations = configurations
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Классификация (LIB)")
        self.setMinimumSize(650, 550)

        layout = QVBoxLayout()

        settings_group = QGroupBox("Параметры классификации")
        form_layout = QFormLayout()

        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems(["Naive Bayes", "SVM", "Random Forest", "KNN"])
        form_layout.addRow("Алгоритм:", self.combo_algorithm)

        self.spin_cv = QSpinBox()
        self.spin_cv.setRange(2, 20)
        self.spin_cv.setValue(5)
        form_layout.addRow("Количество фолдов CV:", self.spin_cv)

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

        self.thread = ClassificationLibThread(
            self.dirname, self.morph, self.configurations,
            self.combo_algorithm.currentText(),
            self.spin_cv.value()
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(lambda t: self.text_result.append(t))
        self.thread.finished_signal.connect(self.onFinished)
        self.thread.error_signal.connect(self.onError)
        self.thread.start()

    def onFinished(self, result):
        self.progress_bar.setValue(100)
        self.btn_start.setEnabled(True)

        self.text_result.append(f"\n=== Результаты классификации ({result['algorithm']}) ===\n")
        self.text_result.append(f"Кросс-валидация: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        self.text_result.append(f"Точность на обучающих данных: {result['accuracy']:.2%}\n")
        self.text_result.append("Отчёт классификации:")
        self.text_result.append(result['report'])

        output_dir = self.configurations.get("output_files_directory", "output_files")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "classification_lib_result.csv")

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
