#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль предобработки текстов.
Содержит классы и функции для загрузки, токенизации, нормализации,
удаления стоп-слов и расчёта частот слов в текстах.
"""

import codecs
import os
import re
from collections import Counter


class TextData:
    """Класс для хранения данных одного текстового документа."""

    def __init__(self):
        self.full_filename = ""
        self.short_filename = ""
        self.text = ""
        self.sentences = []              # Исходные предложения (строки)
        self.tokenized_sentences = []    # Токенизированные предложения (списки слов)
        self.stop_words_pass_sentences = []  # После удаления стоп-слов
        self.normalized_sentences = []   # После нормализации (лемматизации)
        self.register_pass_centences = []  # После приведения регистра
        self.word_frequency = {}         # Частоты слов


def writeStringToFile(string, filename):
    """Запись строки в файл."""
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(string)


def loadInputFilesFromList(filenames):
    """Загрузка текстовых файлов из списка путей."""
    texts = []
    for filename in filenames:
        text_data = TextData()
        text_data.full_filename = filename

        # Пробуем разные кодировки
        content = ""
        for encoding in ['utf-8', 'cp1251', 'latin-1']:
            try:
                with codecs.open(filename, 'r', encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        text_data.text = content

        # Разбиваем на предложения
        raw_sentences = re.split(r'(?<=[.!?])\s+', content)
        text_data.sentences = [s.strip() for s in raw_sentences if s.strip()]

        texts.append(text_data)

    return texts


def tokenizeTextData(texts):
    """Токенизация предложений во всех текстах."""
    for text in texts:
        text.tokenized_sentences = []
        for sentence in text.sentences:
            words = re.findall(r'[а-яА-ЯёЁa-zA-Z0-9]+', sentence)
            if words:
                text.tokenized_sentences.append(words)
    return texts


def removeStopWordsInTexts(texts, morph, configurations):
    """Удаление стоп-слов из токенизированных текстов."""
    stop_words_filename = configurations.get("stop_words_filename", "sources/russian_stop_words.txt")

    stop_words = set()
    try:
        with open(stop_words_filename, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stop_words.add(word)
    except FileNotFoundError:
        print(f"Файл стоп-слов не найден: {stop_words_filename}")

    log_string = ""

    for text in texts:
        text.stop_words_pass_sentences = []
        log_string += f"\nФайл: {text.full_filename}\n"

        for sentence in text.tokenized_sentences:
            filtered = [word for word in sentence if word.lower() not in stop_words]
            text.stop_words_pass_sentences.append(filtered)

            removed = [word for word in sentence if word.lower() in stop_words]
            if removed:
                log_string += f" Удалены: {', '.join(removed)}\n"

    return texts, log_string


def normalizeTexts(texts, morph):
    """Нормализация (лемматизация) слов в текстах."""
    log_string = ""

    for text in texts:
        text.normalized_sentences = []
        log_string += f"\nФайл: {text.full_filename}\n"

        for sentence in text.stop_words_pass_sentences:
            normalized = []
            for word in sentence:
                parsed = morph.parse(word)
                if parsed:
                    normal_form = parsed[0].normal_form
                    normalized.append(normal_form)
                    if normal_form != word.lower():
                        log_string += f" {word} -> {normal_form}\n"
                else:
                    normalized.append(word)
            text.normalized_sentences.append(normalized)

    return texts, log_string


def fixRegisterInTexts(texts, morph):
    """Приведение регистра слов (все с маленькой буквы, кроме ФИО)."""
    log_string = ""

    for text in texts:
        text.register_pass_centences = []
        log_string += f"\nФайл: {text.full_filename}\n"

        for sentence in text.normalized_sentences:
            fixed = []
            for word in sentence:
                parsed = morph.parse(word)
                is_name = False
                if parsed:
                    tag = parsed[0].tag
                    # Проверяем, является ли слово именем собственным
                    if 'Name' in str(tag) or 'Surn' in str(tag) or 'Patr' in str(tag):
                        is_name = True

                if is_name:
                    fixed.append(word.capitalize())
                else:
                    lower_word = word.lower()
                    if lower_word != word:
                        log_string += f" {word} -> {lower_word}\n"
                    fixed.append(lower_word)

            text.register_pass_centences.append(fixed)

    return texts, log_string


def calculateWordsFrequencyInTexts(texts):
    """Подсчёт частоты слов во всех текстах."""
    log_string = "Слово;Частота\n"

    global_counter = Counter()

    for text in texts:
        text.word_frequency = Counter()
        for sentence in text.register_pass_centences:
            for word in sentence:
                text.word_frequency[word] += 1
                global_counter[word] += 1

    # Формируем лог (сортировка по частоте)
    for word, freq in global_counter.most_common():
        log_string += f"{word};{freq}\n"

    return texts, log_string
