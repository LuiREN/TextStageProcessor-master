#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль обучения и использования моделей Word2Vec."""

import os
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QComboBox, QLineEdit, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from gensim.models import Word2Vec

from sources.utils import (
    readTextFile, readStopWords, preprocessText,
    tokenize, lemmatize, sentenceTokenize
)


class Word2VecThread(QThread):
    """Поток для обучения Word2Vec."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, filename, morph, configurations, params, input_dir):
        super().__init__()
        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.params = params
        self.input_dir = input_dir

    def run(self):
        try:
            # Если файл - модель, загружаем её
            if self.filename.endswith('.model'):
                self.log_signal.emit("Загрузка существующей модели...")
                model = Word2Vec.load(self.filename)
                self.progress.emit(100)
                self.finished_signal.emit({
                    'model': model,
                    'mode': 'loaded',
                    'vocab_size': len(model.wv)
                })
                return

            stop_words_file = self.configurations.get("stop_words_filename", "sources/russian_stop_words.txt")
            stop_words = readStopWords(stop_words_file)

            self.log_signal.emit("Чтение и предобработка текста...")
            text = readTextFile(self.filename)
            sentences_text = sentenceTokenize(text)

            self.progress.emit(20)

            # Подготовка предложений для обучения
            sentences = []
            for i, sent_text in enumerate(sentences_text):
                tokens = preprocessText(sent_text, self.morph, stop_words)
                if tokens:
                    sentences.append(tokens)
                if i % 100 == 0:
                    self.progress.emit(20 + int((i + 1) / len(sentences_text) * 30))

            self.log_signal.emit(f"Подготовлено {len(sentences)} предложений для обучения")
            self.progress.emit(50)

            if len(sentences) < 2:
                self.error_signal.emit("Недостаточно предложений для обучения модели")
                return

            # Обучение модели
            self.log_signal.emit("Обучение модели Word2Vec...")
            model = Word2Vec(
                sentences=sentences,
                vector_size=self.params['vector_size'],
                window=self.params['window'],
                min_count=self.params['min_count'],
                workers=self.params['workers'],
                epochs=self.params['epochs'],
                sg=1 if self.params['algorithm'] == 'Skip-gram' else 0
            )

            self.progress.emit(90)

            # Сохранение модели
            output_dir = self.configurations.get("output_files_directory", "output_files")
            w2v_dir = os.path.join(output_dir, "Word2Vec")
            if not os.path.exists(w2v_dir):
                os.makedirs(w2v_dir)

            model_path = os.path.join(w2v_dir, "word2vec.model")
            model.save(model_path)
            self.log_signal.emit(f"Модель сохранена: {model_path}")

            self.progress.emit(100)
            self.finished_signal.emit({
                'model': model,
                'mode': 'trained',
                'vocab_size': len(model.wv),
                'model_path': model_path
            })

        except Exception as e:
            self.error_signal.emit(str(e))


class DialogWord2VecMaker(QDialog):
    """Диалог обучения и использования Word2Vec."""

    def __init__(self, input_dir, filename, morph, configurations, parent=None):
        super().__init__(parent)
        self.input_dir = input_dir
        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.model = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Word2Vec")
        self.setMinimumSize(700, 650)

        layout = QVBoxLayout()

        # Параметры обучения
        train_group = QGroupBox("Параметры обучения")
        form_layout = QFormLayout()

        self.spin_vector_size = QSpinBox()
        self.spin_vector_size.setRange(50, 500)
        self.spin_vector_size.setValue(100)
        form_layout.addRow("Размер вектора:", self.spin_vector_size)

        self.spin_window = QSpinBox()
        self.spin_window.setRange(1, 20)
        self.spin_window.setValue(5)
        form_layout.addRow("Размер окна:", self.spin_window)

        self.spin_min_count = QSpinBox()
        self.spin_min_count.setRange(1, 100)
        self.spin_min_count.setValue(2)
        form_layout.addRow("Мин. частота слова:", self.spin_min_count)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 100)
        self.spin_epochs.setValue(10)
        form_layout.addRow("Количество эпох:", self.spin_epochs)

        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems(["CBOW", "Skip-gram"])
        form_layout.addRow("Алгоритм:", self.combo_algorithm)

        train_group.setLayout(form_layout)
        layout.addWidget(train_group)

        # Прогресс
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Кнопка обучения
        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("Обучить модель / Загрузить")
        self.btn_train.clicked.connect(self.startTraining)
        btn_layout.addWidget(self.btn_train)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        # Поиск похожих слов
        query_group = QGroupBox("Поиск похожих слов")
        query_layout = QHBoxLayout()
        self.edit_word = QLineEdit()
        self.edit_word.setPlaceholderText("Введите слово")
        query_layout.addWidget(self.edit_word)

        self.spin_topn = QSpinBox()
        self.spin_topn.setRange(1, 50)
        self.spin_topn.setValue(10)
        query_layout.addWidget(QLabel("Кол-во:"))
        query_layout.addWidget(self.spin_topn)

        self.btn_search = QPushButton("Найти")
        self.btn_search.clicked.connect(self.searchSimilar)
        self.btn_search.setEnabled(False)
        query_layout.addWidget(self.btn_search)

        query_group.setLayout(query_layout)
        layout.addWidget(query_group)

        # Результаты
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        layout.addWidget(self.text_result)

        self.setLayout(layout)

    def startTraining(self):
        self.btn_train.setEnabled(False)
        self.text_result.clear()
        self.progress_bar.setValue(0)

        params = {
            'vector_size': self.spin_vector_size.value(),
            'window': self.spin_window.value(),
            'min_count': self.spin_min_count.value(),
            'epochs': self.spin_epochs.value(),
            'algorithm': self.combo_algorithm.currentText(),
            'workers': 4
        }

        self.thread = Word2VecThread(
            self.filename, self.morph, self.configurations,
            params, self.input_dir
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(lambda t: self.text_result.append(t))
        self.thread.finished_signal.connect(self.onFinished)
        self.thread.error_signal.connect(self.onError)
        self.thread.start()

    def onFinished(self, result):
        self.progress_bar.setValue(100)
        self.btn_train.setEnabled(True)
        self.btn_search.setEnabled(True)
        self.model = result['model']

        mode_text = "обучена" if result['mode'] == 'trained' else "загружена"
        self.text_result.append(f"\nМодель {mode_text} успешно!")
        self.text_result.append(f"Размер словаря: {result['vocab_size']}")

        if 'model_path' in result:
            self.text_result.append(f"Путь к модели: {result['model_path']}")

    def searchSimilar(self):
        if self.model is None:
            QMessageBox.warning(self, "Внимание", "Сначала обучите или загрузите модель")
            return

        word = self.edit_word.text().strip().lower()
        if not word:
            return

        # Лемматизация поискового слова
        parsed = self.morph.parse(word)
        if parsed:
            word_lemma = parsed[0].normal_form
        else:
            word_lemma = word

        try:
            if word_lemma in self.model.wv:
                similar = self.model.wv.most_similar(word_lemma, topn=self.spin_topn.value())
                self.text_result.append(f"\nПохожие на '{word_lemma}':")
                for w, score in similar:
                    self.text_result.append(f"  {w}: {score:.4f}")
            elif word in self.model.wv:
                similar = self.model.wv.most_similar(word, topn=self.spin_topn.value())
                self.text_result.append(f"\nПохожие на '{word}':")
                for w, score in similar:
                    self.text_result.append(f"  {w}: {score:.4f}")
            else:
                self.text_result.append(f"\nСлово '{word}' ('{word_lemma}') не найдено в словаре модели")
        except KeyError as e:
            self.text_result.append(f"\nСлово не найдено в словаре: {e}")

    def onError(self, error_text):
        self.btn_train.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_text}")
