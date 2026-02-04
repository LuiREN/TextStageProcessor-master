#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль обучения и использования моделей FastText через gensim."""

import os
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QComboBox, QLineEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from gensim.models import FastText

from sources.utils import (
    readTextFile, readStopWords, preprocessText,
    tokenize, lemmatize, sentenceTokenize
)


class FastTextThread(QThread):
    """Поток для обучения FastText."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, input_dir, morph, configurations, params, training_file):
        super().__init__()
        self.input_dir = input_dir
        self.morph = morph
        self.configurations = configurations
        self.params = params
        self.training_file = training_file

    def run(self):
        try:
            stop_words_file = self.configurations.get("stop_words_filename", "sources/russian_stop_words.txt")
            stop_words = readStopWords(stop_words_file)

            self.log_signal.emit("Чтение и подготовка данных...")

            sentences = []

            if self.training_file and os.path.isfile(self.training_file):
                text = readTextFile(self.training_file)
                sentences_text = sentenceTokenize(text)
                for sent in sentences_text:
                    tokens = preprocessText(sent, self.morph, stop_words)
                    if tokens:
                        sentences.append(tokens)
            else:
                # Читаем все txt файлы из input_dir
                if os.path.exists(self.input_dir):
                    for fname in os.listdir(self.input_dir):
                        if fname.endswith('.txt'):
                            filepath = os.path.join(self.input_dir, fname)
                            text = readTextFile(filepath)
                            sentences_text = sentenceTokenize(text)
                            for sent in sentences_text:
                                tokens = preprocessText(sent, self.morph, stop_words)
                                if tokens:
                                    sentences.append(tokens)

            self.progress.emit(30)
            self.log_signal.emit(f"Подготовлено {len(sentences)} предложений")

            if len(sentences) < 2:
                self.error_signal.emit("Недостаточно данных для обучения. Добавьте txt файлы в input_files/")
                return

            # Обучение модели
            self.log_signal.emit("Обучение модели FastText (gensim)...")

            model = FastText(
                sentences=sentences,
                vector_size=self.params['vector_size'],
                window=self.params['window'],
                min_count=self.params['min_count'],
                workers=4,
                epochs=self.params['epochs'],
                sg=1 if self.params['algorithm'] == 'Skip-gram' else 0,
                min_n=self.params['min_n'],
                max_n=self.params['max_n']
            )

            self.progress.emit(90)

            # Сохранение
            output_dir = self.configurations.get("output_files_directory", "output_files")
            ft_dir = os.path.join(output_dir, "FastText")
            if not os.path.exists(ft_dir):
                os.makedirs(ft_dir)

            model_path = os.path.join(ft_dir, "fasttext.model")
            model.save(model_path)
            self.log_signal.emit(f"Модель сохранена: {model_path}")

            self.progress.emit(100)
            self.finished_signal.emit({
                'model': model,
                'vocab_size': len(model.wv),
                'model_path': model_path
            })

        except Exception as e:
            self.error_signal.emit(str(e))


class DialogFastTextMaker(QDialog):
    """Диалог обучения и использования FastText."""

    def __init__(self, input_dir, morph, configurations, parent=None):
        super().__init__(parent)
        self.input_dir = input_dir
        self.morph = morph
        self.configurations = configurations
        self.model = None
        self.training_file = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("FastText")
        self.setMinimumSize(700, 650)

        layout = QVBoxLayout()

        # Выбор файла
        file_group = QGroupBox("Данные для обучения")
        file_layout = QHBoxLayout()
        self.label_file = QLabel("Файл не выбран (будут использованы все txt из input_files/)")
        file_layout.addWidget(self.label_file)
        btn_select = QPushButton("Выбрать файл")
        btn_select.clicked.connect(self.selectFile)
        file_layout.addWidget(btn_select)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Параметры
        params_group = QGroupBox("Параметры обучения")
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

        self.spin_min_n = QSpinBox()
        self.spin_min_n.setRange(1, 10)
        self.spin_min_n.setValue(3)
        form_layout.addRow("Мин. длина n-грамм:", self.spin_min_n)

        self.spin_max_n = QSpinBox()
        self.spin_max_n.setRange(3, 20)
        self.spin_max_n.setValue(6)
        form_layout.addRow("Макс. длина n-грамм:", self.spin_max_n)

        params_group.setLayout(form_layout)
        layout.addWidget(params_group)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("Обучить модель")
        self.btn_train.clicked.connect(self.startTraining)
        btn_layout.addWidget(self.btn_train)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        # Поиск
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

        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        layout.addWidget(self.text_result)

        self.setLayout(layout)

    def selectFile(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Выберите текстовый файл", self.input_dir,
            "Text files (*.txt);;All files (*.*)"
        )
        if filename:
            self.training_file = filename
            self.label_file.setText(os.path.basename(filename))

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
            'min_n': self.spin_min_n.value(),
            'max_n': self.spin_max_n.value()
        }

        self.thread = FastTextThread(
            self.input_dir, self.morph, self.configurations,
            params, self.training_file
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

        self.text_result.append(f"\nМодель обучена успешно!")
        self.text_result.append(f"Размер словаря: {result['vocab_size']}")
        self.text_result.append(f"Путь к модели: {result['model_path']}")

    def searchSimilar(self):
        if self.model is None:
            QMessageBox.warning(self, "Внимание", "Сначала обучите модель")
            return

        word = self.edit_word.text().strip().lower()
        if not word:
            return

        parsed = self.morph.parse(word)
        if parsed:
            word_lemma = parsed[0].normal_form
        else:
            word_lemma = word

        try:
            # FastText может работать с OOV словами через n-граммы
            similar = self.model.wv.most_similar(word_lemma, topn=self.spin_topn.value())
            self.text_result.append(f"\nПохожие на '{word_lemma}':")
            for w, score in similar:
                self.text_result.append(f"  {w}: {score:.4f}")
        except KeyError as e:
            self.text_result.append(f"\nНе удалось найти похожие слова: {e}")

    def onError(self, error_text):
        self.btn_train.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_text}")
