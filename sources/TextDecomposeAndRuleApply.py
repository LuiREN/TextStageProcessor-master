#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль анализа и применения правил вывода предложений."""

import os
import re
import numpy as np
from collections import Counter

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QSpinBox, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from sources.utils import (
    readTextFile, readStopWords, preprocessText,
    tokenize, lemmatize, sentenceTokenize, writeResultToCSV
)


class DRAThread(QThread):
    """Поток для анализа и применения правил."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, filenames, morph, configurations, min_sentence_len, use_lemmatization):
        super().__init__()
        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.min_sentence_len = min_sentence_len
        self.use_lemmatization = use_lemmatization

    def run(self):
        try:
            stop_words_file = self.configurations.get("stop_words_filename", "sources/russian_stop_words.txt")
            stop_words = readStopWords(stop_words_file)

            all_results = []

            for file_idx, filename in enumerate(self.filenames):
                self.log_signal.emit(f"Обработка файла: {os.path.basename(filename)}")

                text = readTextFile(filename)
                sentences = sentenceTokenize(text)

                doc_result = {
                    'filename': os.path.basename(filename),
                    'total_sentences': len(sentences),
                    'sentences_analysis': []
                }

                for i, sentence in enumerate(sentences):
                    tokens = tokenize(sentence)
                    if len(tokens) < self.min_sentence_len:
                        continue

                    # Морфологический анализ каждого слова
                    word_analysis = []
                    pos_tags = []
                    for token in tokens:
                        parsed = self.morph.parse(token)
                        if parsed:
                            p = parsed[0]
                            pos = str(p.tag.POS) if p.tag.POS else "UNKN"
                            pos_tags.append(pos)
                            word_analysis.append({
                                'word': token,
                                'normal_form': p.normal_form,
                                'pos': pos,
                                'tag': str(p.tag)
                            })

                    # Правила извлечения
                    extracted_info = self._applyRules(word_analysis, pos_tags, sentence)

                    # Фильтрация стоп-слов
                    if self.use_lemmatization:
                        key_words = [wa['normal_form'] for wa in word_analysis
                                     if wa['normal_form'] not in stop_words
                                     and wa['pos'] in ('NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN')]
                    else:
                        key_words = [wa['word'] for wa in word_analysis
                                     if wa['word'].lower() not in stop_words]

                    sent_result = {
                        'sentence': sentence,
                        'word_count': len(tokens),
                        'pos_distribution': Counter(pos_tags),
                        'key_words': key_words,
                        'extracted_info': extracted_info,
                        'word_analysis': word_analysis
                    }
                    doc_result['sentences_analysis'].append(sent_result)

                all_results.append(doc_result)
                self.progress.emit(int((file_idx + 1) / len(self.filenames) * 100))

            self.finished_signal.emit(all_results)

        except Exception as e:
            self.error_signal.emit(str(e))

    def _applyRules(self, word_analysis, pos_tags, sentence):
        """Применение правил вывода."""
        extracted = []

        # Правило 1: Именные группы (прил + сущ)
        for i in range(len(pos_tags) - 1):
            if pos_tags[i] in ('ADJF', 'ADJS') and pos_tags[i + 1] == 'NOUN':
                extracted.append({
                    'rule': 'Именная группа (прил+сущ)',
                    'text': f"{word_analysis[i]['word']} {word_analysis[i + 1]['word']}",
                    'normal': f"{word_analysis[i]['normal_form']} {word_analysis[i + 1]['normal_form']}"
                })

        # Правило 2: Субъект-предикат (сущ + глагол)
        for i in range(len(pos_tags) - 1):
            if pos_tags[i] == 'NOUN' and pos_tags[i + 1] in ('VERB', 'INFN'):
                extracted.append({
                    'rule': 'Субъект-предикат',
                    'text': f"{word_analysis[i]['word']} {word_analysis[i + 1]['word']}",
                    'normal': f"{word_analysis[i]['normal_form']} {word_analysis[i + 1]['normal_form']}"
                })

        # Правило 3: Тройные именные группы (прил + прил + сущ)
        for i in range(len(pos_tags) - 2):
            if (pos_tags[i] in ('ADJF', 'ADJS') and
                pos_tags[i + 1] in ('ADJF', 'ADJS') and
                pos_tags[i + 2] == 'NOUN'):
                extracted.append({
                    'rule': 'Тройная именная группа',
                    'text': f"{word_analysis[i]['word']} {word_analysis[i + 1]['word']} {word_analysis[i + 2]['word']}",
                    'normal': f"{word_analysis[i]['normal_form']} {word_analysis[i + 1]['normal_form']} {word_analysis[i + 2]['normal_form']}"
                })

        return extracted


class DialogConfigDRA(QDialog):
    """Диалог анализа и применения правил вывода."""

    def __init__(self, filenames, morph, configurations, parent=None):
        super().__init__(parent)
        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Анализ и правила вывода предложений")
        self.setMinimumSize(700, 600)

        layout = QVBoxLayout()

        settings_group = QGroupBox("Параметры анализа")
        form_layout = QFormLayout()

        self.spin_min_len = QSpinBox()
        self.spin_min_len.setRange(1, 50)
        self.spin_min_len.setValue(3)
        form_layout.addRow("Мин. длина предложения (слов):", self.spin_min_len)

        self.check_lemma = QCheckBox("Использовать лемматизацию")
        self.check_lemma.setChecked(True)
        form_layout.addRow(self.check_lemma)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        info_label = QLabel(f"Выбрано файлов: {len(self.filenames)}")
        layout.addWidget(info_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Начать анализ")
        self.btn_start.clicked.connect(self.startAnalysis)
        btn_layout.addWidget(self.btn_start)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        layout.addWidget(self.text_result)

        self.setLayout(layout)

    def startAnalysis(self):
        self.btn_start.setEnabled(False)
        self.text_result.clear()
        self.progress_bar.setValue(0)

        self.thread = DRAThread(
            self.filenames, self.morph, self.configurations,
            self.spin_min_len.value(),
            self.check_lemma.isChecked()
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(lambda t: self.text_result.append(t))
        self.thread.finished_signal.connect(self.onFinished)
        self.thread.error_signal.connect(self.onError)
        self.thread.start()

    def onFinished(self, results):
        self.progress_bar.setValue(100)
        self.btn_start.setEnabled(True)

        output_dir = self.configurations.get("output_files_directory", "output_files")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for doc_result in results:
            self.text_result.append(f"\n=== {doc_result['filename']} ===")
            self.text_result.append(f"Всего предложений: {doc_result['total_sentences']}")
            self.text_result.append(f"Обработано: {len(doc_result['sentences_analysis'])}\n")

            all_extracted = []
            for sent in doc_result['sentences_analysis']:
                if sent['extracted_info']:
                    self.text_result.append(f"Предложение: {sent['sentence'][:80]}...")
                    for info in sent['extracted_info']:
                        self.text_result.append(f"  [{info['rule']}]: {info['text']} -> {info['normal']}")
                        all_extracted.append(info)
                    self.text_result.append("")

            self.text_result.append(f"Всего извлечено конструкций: {len(all_extracted)}")

            # Сохранение
            output_file = os.path.join(output_dir, f"dra_{doc_result['filename']}.csv")
            headers = ["Правило", "Исходный текст", "Нормальная форма"]
            rows = [[info['rule'], info['text'], info['normal']] for info in all_extracted]
            writeResultToCSV(output_file, headers, rows)
            self.text_result.append(f"Сохранено в: {output_file}")

    def onError(self, error_text):
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_text}")
