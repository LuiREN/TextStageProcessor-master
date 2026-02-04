#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль расчёта критериев хи-квадрат для выделения термов."""

import os
import csv
import numpy as np
from collections import Counter

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from sources.utils import (
    readTextFile, readStopWords, preprocessText,
    tokenize, lemmatize, writeResultToCSV
)


class XiSquareThread(QThread):
    """Поток для расчёта хи-квадрат."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, filename, morph, configurations, top_n, min_freq):
        super().__init__()
        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.top_n = top_n
        self.min_freq = min_freq

    def run(self):
        try:
            stop_words_file = self.configurations.get("stop_words_filename", "sources/russian_stop_words.txt")
            stop_words = readStopWords(stop_words_file)

            self.log_signal.emit("Чтение CSV файла...")

            # Чтение CSV с документами и категориями
            documents = []
            categories = []

            with open(self.filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f, delimiter=';')
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        documents.append(row[0])
                        categories.append(row[1])
                    elif len(row) == 1:
                        documents.append(row[0])
                        categories.append("default")

            if not documents:
                # Попробуем читать как простой текст
                text = readTextFile(self.filename)
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                documents = paragraphs
                categories = ["default"] * len(paragraphs)

            self.progress.emit(20)
            self.log_signal.emit(f"Загружено {len(documents)} документов")

            # Предобработка
            self.log_signal.emit("Предобработка текстов...")
            processed_docs = []
            for i, doc in enumerate(documents):
                tokens = preprocessText(doc, self.morph, stop_words)
                processed_docs.append(tokens)
                self.progress.emit(20 + int((i + 1) / len(documents) * 30))

            # Подсчёт статистик для хи-квадрат
            self.log_signal.emit("Расчёт статистики хи-квадрат...")

            unique_categories = list(set(categories))
            all_terms = Counter()
            for doc in processed_docs:
                all_terms.update(set(doc))

            # Фильтрация по минимальной частоте
            terms = [t for t, freq in all_terms.items() if freq >= self.min_freq]

            N = len(documents)
            results = []

            for term_idx, term in enumerate(terms):
                chi2_total = 0

                for category in unique_categories:
                    # A: документы с термом в категории
                    # B: документы с термом не в категории
                    # C: документы без терма в категории
                    # D: документы без терма не в категории
                    A = B = C = D = 0

                    for i, doc in enumerate(processed_docs):
                        has_term = term in doc
                        is_cat = categories[i] == category

                        if has_term and is_cat:
                            A += 1
                        elif has_term and not is_cat:
                            B += 1
                        elif not has_term and is_cat:
                            C += 1
                        else:
                            D += 1

                    # Формула хи-квадрат
                    numerator = (A * D - B * C) ** 2 * N
                    denominator = (A + B) * (C + D) * (A + C) * (B + D)

                    if denominator > 0:
                        chi2 = numerator / denominator
                    else:
                        chi2 = 0

                    chi2_total += chi2

                results.append({
                    'term': term,
                    'chi2': chi2_total,
                    'frequency': all_terms[term]
                })

                if term_idx % 50 == 0:
                    self.progress.emit(50 + int(term_idx / len(terms) * 40))

            # Сортировка по хи-квадрат
            results.sort(key=lambda x: -x['chi2'])
            top_results = results[:self.top_n]

            self.progress.emit(100)
            self.finished_signal.emit({
                'results': top_results,
                'total_terms': len(terms),
                'total_docs': N,
                'categories': unique_categories
            })

        except Exception as e:
            self.error_signal.emit(str(e))


class DialogXiSquare(QDialog):
    """Диалог расчёта хи-квадрат для выделения термов."""

    def __init__(self, filename, morph, configurations, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Расчёт критериев хи-квадрат")
        self.setMinimumSize(700, 600)

        layout = QVBoxLayout()

        settings_group = QGroupBox("Параметры")
        form_layout = QFormLayout()

        self.spin_top = QSpinBox()
        self.spin_top.setRange(10, 1000)
        self.spin_top.setValue(50)
        form_layout.addRow("Кол-во топ термов:", self.spin_top)

        self.spin_min_freq = QSpinBox()
        self.spin_min_freq.setRange(1, 100)
        self.spin_min_freq.setValue(2)
        form_layout.addRow("Мин. частота терма:", self.spin_min_freq)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        info_label = QLabel(f"Файл: {os.path.basename(self.filename)}")
        layout.addWidget(info_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Рассчитать")
        self.btn_start.clicked.connect(self.startCalculation)
        btn_layout.addWidget(self.btn_start)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        layout.addWidget(self.text_result)

        self.setLayout(layout)

    def startCalculation(self):
        self.btn_start.setEnabled(False)
        self.text_result.clear()
        self.progress_bar.setValue(0)

        self.thread = XiSquareThread(
            self.filename, self.morph, self.configurations,
            self.spin_top.value(),
            self.spin_min_freq.value()
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(lambda t: self.text_result.append(t))
        self.thread.finished_signal.connect(self.onFinished)
        self.thread.error_signal.connect(self.onError)
        self.thread.start()

    def onFinished(self, result):
        self.progress_bar.setValue(100)
        self.btn_start.setEnabled(True)

        self.text_result.append(f"\n=== Результаты расчёта хи-квадрат ===\n")
        self.text_result.append(f"Всего документов: {result['total_docs']}")
        self.text_result.append(f"Всего уникальных термов: {result['total_terms']}")
        self.text_result.append(f"Категорий: {len(result['categories'])}\n")

        self.text_result.append(f"Топ-{len(result['results'])} термов по хи-квадрат:\n")
        self.text_result.append(f"{'№':<5}{'Терм':<30}{'χ²':<15}{'Частота':<10}")
        self.text_result.append("-" * 60)

        for i, r in enumerate(result['results']):
            self.text_result.append(f"{i + 1:<5}{r['term']:<30}{r['chi2']:<15.4f}{r['frequency']:<10}")

        output_dir = self.configurations.get("output_files_directory", "output_files")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "xi_square_result.csv")

        headers = ["Терм", "Хи-квадрат", "Частота"]
        rows = [[r['term'], f"{r['chi2']:.4f}", r['frequency']] for r in result['results']]
        writeResultToCSV(output_file, headers, rows)
        self.text_result.append(f"\nРезультаты сохранены в: {output_file}")

    def onError(self, error_text):
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_text}")
