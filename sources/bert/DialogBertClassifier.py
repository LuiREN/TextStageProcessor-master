#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Модуль классификации текстов с использованием BERT."""

import os
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from sources.utils import readTextFile, writeResultToCSV


class BertClassificationThread(QThread):
    """Поток для классификации с BERT."""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, dirname, morph, configurations, params):
        super().__init__()
        self.dirname = dirname
        self.morph = morph
        self.configurations = configurations
        self.params = params

    def run(self):
        try:
            self.log_signal.emit("Импорт библиотек PyTorch и Transformers...")
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                from torch.utils.data import DataLoader, Dataset
            except ImportError as e:
                self.error_signal.emit(
                    f"Не удалось импортировать необходимые библиотеки: {e}\n"
                    "Убедитесь, что установлены torch и transformers:\n"
                    "pip install torch transformers"
                )
                return

            self.progress.emit(10)

            # Чтение данных
            self.log_signal.emit("Чтение данных...")
            texts = []
            labels = []
            label_names = []
            doc_names = []

            subdirs = [d for d in os.listdir(self.dirname)
                       if os.path.isdir(os.path.join(self.dirname, d))]

            if len(subdirs) < 2:
                self.error_signal.emit("Необходимо минимум 2 класса (поддиректории)")
                return

            label_map = {}
            for idx, subdir in enumerate(sorted(subdirs)):
                label_map[subdir] = idx
                label_names.append(subdir)
                class_dir = os.path.join(self.dirname, subdir)
                files = [f for f in os.listdir(class_dir) if f.endswith('.txt')]
                for filename in files:
                    filepath = os.path.join(class_dir, filename)
                    text = readTextFile(filepath)
                    texts.append(text[:512])  # Ограничение длины для BERT
                    labels.append(idx)
                    doc_names.append(filename)

            self.progress.emit(20)
            self.log_signal.emit(f"Загружено {len(texts)} документов, {len(label_names)} классов")

            # Загрузка токенизатора и модели
            model_name = self.params['model_name']
            self.log_signal.emit(f"Загрузка модели {model_name}...")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_signal.emit(f"Устройство: {device}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.progress.emit(40)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(label_names)
            )
            model.to(device)
            self.progress.emit(50)

            # Токенизация
            self.log_signal.emit("Токенизация текстов...")
            encodings = tokenizer(
                texts, truncation=True, padding=True,
                max_length=self.params['max_length'],
                return_tensors='pt'
            )
            labels_tensor = torch.tensor(labels)

            self.progress.emit(60)

            # Простая классификация (fine-tuning)
            self.log_signal.emit("Обучение классификатора...")

            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.params['learning_rate'])

            batch_size = self.params['batch_size']
            n_epochs = self.params['epochs']
            n_samples = len(texts)

            for epoch in range(n_epochs):
                total_loss = 0
                indices = torch.randperm(n_samples)

                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch_idx = indices[start:end]

                    batch_input_ids = encodings['input_ids'][batch_idx].to(device)
                    batch_attention = encodings['attention_mask'][batch_idx].to(device)
                    batch_labels = labels_tensor[batch_idx].to(device)

                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention,
                        labels=batch_labels
                    )
                    loss = outputs.loss
                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                avg_loss = total_loss / max(1, n_samples // batch_size)
                self.log_signal.emit(f"Эпоха {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
                self.progress.emit(60 + int((epoch + 1) / n_epochs * 30))

            # Предсказания
            self.log_signal.emit("Получение предсказаний...")
            model.eval()
            predictions = []

            with torch.no_grad():
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch_input_ids = encodings['input_ids'][start:end].to(device)
                    batch_attention = encodings['attention_mask'][start:end].to(device)

                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention
                    )
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                    predictions.extend(preds)

            correct = sum(1 for p, a in zip(predictions, labels) if p == a)
            accuracy = correct / len(labels) if labels else 0

            result = {
                'predictions': [label_names[p] for p in predictions],
                'actual': [label_names[a] for a in labels],
                'doc_names': doc_names,
                'accuracy': accuracy,
                'correct': correct,
                'total': len(labels),
                'label_names': label_names,
                'model_name': model_name
            }

            self.progress.emit(100)
            self.finished_signal.emit(result)

        except Exception as e:
            self.error_signal.emit(str(e))


class DialogBertClassifier(QDialog):
    """Диалог классификации с использованием BERT."""

    def __init__(self, dirname, morph, configurations, parent=None):
        super().__init__(parent)
        self.dirname = dirname
        self.morph = morph
        self.configurations = configurations
        self.initUI()

    def initUI(self):
        self.setWindowTitle("BERT Классификация")
        self.setMinimumSize(700, 600)

        layout = QVBoxLayout()

        settings_group = QGroupBox("Параметры BERT")
        form_layout = QFormLayout()

        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "bert-base-multilingual-cased",
            "DeepPavlov/rubert-base-cased",
            "ai-forever/ruBert-base"
        ])
        form_layout.addRow("Модель:", self.combo_model)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 20)
        self.spin_epochs.setValue(3)
        form_layout.addRow("Количество эпох:", self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(8)
        form_layout.addRow("Размер батча:", self.spin_batch)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.000001, 0.01)
        self.spin_lr.setValue(0.00002)
        self.spin_lr.setDecimals(6)
        self.spin_lr.setSingleStep(0.000001)
        form_layout.addRow("Learning rate:", self.spin_lr)

        self.spin_max_len = QSpinBox()
        self.spin_max_len.setRange(32, 512)
        self.spin_max_len.setValue(128)
        form_layout.addRow("Макс. длина токенов:", self.spin_max_len)

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

        params = {
            'model_name': self.combo_model.currentText(),
            'epochs': self.spin_epochs.value(),
            'batch_size': self.spin_batch.value(),
            'learning_rate': self.spin_lr.value(),
            'max_length': self.spin_max_len.value()
        }

        self.thread = BertClassificationThread(
            self.dirname, self.morph, self.configurations, params
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(lambda t: self.text_result.append(t))
        self.thread.finished_signal.connect(self.onFinished)
        self.thread.error_signal.connect(self.onError)
        self.thread.start()

    def onFinished(self, result):
        self.progress_bar.setValue(100)
        self.btn_start.setEnabled(True)

        self.text_result.append(f"\n=== Результаты BERT классификации ===")
        self.text_result.append(f"Модель: {result['model_name']}")
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
        output_file = os.path.join(output_dir, "bert_classification_result.csv")

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
