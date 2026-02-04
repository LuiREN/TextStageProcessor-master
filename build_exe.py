#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт сборки exe-файла приложения "Этапный текстовый процессор"
с использованием PyInstaller.

Использование:
    python build_exe.py

Результат: папка dist/TextStageProcessor/ с исполняемым файлом
"""

import os
import sys
import subprocess
import shutil


def get_pymorphy_dict_path():
    """Определяем путь к словарям pymorphy."""
    try:
        import pymorphy2_dicts_ru
        return os.path.dirname(pymorphy2_dicts_ru.__file__)
    except ImportError:
        pass
    try:
        import pymorphy3_dicts_ru
        return os.path.dirname(pymorphy3_dicts_ru.__file__)
    except ImportError:
        pass
    return None


def build():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    # Определяем пути к данным
    pymorphy_path = get_pymorphy_dict_path()

    # Формируем список data-файлов
    datas = [
        ('configuration.cfg', '.'),
        ('sources/russian_stop_words.txt', 'sources'),
    ]

    if pymorphy_path:
        datas.append((pymorphy_path, 'pymorphy2_dicts_ru' if 'pymorphy2' in pymorphy_path else 'pymorphy3_dicts_ru'))

    # Формируем аргументы для PyInstaller
    separator = ';' if sys.platform == 'win32' else ':'

    args = [
        'pyinstaller',
        '--name=TextStageProcessor',
        '--noconfirm',
        '--clean',
    ]

    # Добавляем data-файлы
    for src, dst in datas:
        args.append(f'--add-data={src}{separator}{dst}')

    # Hidden imports для корректной работы
    hidden_imports = [
        'pymorphy2',
        'pymorphy2.units',
        'pymorphy2.units.by_analogy',
        'pymorphy2.units.by_hyphen',
        'pymorphy2.units.by_lookup',
        'pymorphy2.units.by_shape',
        'pymorphy2.units.unkn',
        'pymorphy2_dicts_ru',
        'pymorphy3',
        'pymorphy3_dicts_ru',
        'dawg_python',
        'dawg2_python',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'sklearn.neighbors._partition_nodes',
        'sklearn.tree._utils',
        'gensim.models.word2vec',
        'gensim.models.fasttext',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.sip',
        'sources',
        'sources.utils',
        'sources.TextClasterization',
        'sources.TextClastering',
        'sources.TextClassification',
        'sources.TextClassificationLib',
        'sources.TextLSA',
        'sources.TextDecomposeAndRuleApply',
        'sources.XiSquare',
        'sources.AnnotationMaker',
        'sources.Word2VecNew',
        'sources.Word2VecNew.DialogWord2VecMaker',
        'sources.FastText',
        'sources.FastText.DialogFastTextMaker',
        'sources.bert',
        'sources.bert.DialogBertClassifier',
    ]

    for imp in hidden_imports:
        args.append(f'--hidden-import={imp}')

    # Исключаем тяжёлые библиотеки (torch ~2GB, transformers ~500MB)
    # BERT функциональность потребует отдельной установки
    excludes = [
        'torch',
        'transformers',
        'torchaudio',
        'torchvision',
        'tensorflow',
        'keras',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'docutils',
    ]

    for exc in excludes:
        args.append(f'--exclude-module={exc}')

    # Точка входа
    args.append('stage_text_processor.py')

    print("=" * 60)
    print("Сборка TextStageProcessor")
    print("=" * 60)
    print(f"Директория проекта: {project_dir}")
    print(f"Словари pymorphy: {pymorphy_path}")
    print(f"Платформа: {sys.platform}")
    print("=" * 60)
    print("Запуск PyInstaller...")
    print(" ".join(args))
    print("=" * 60)

    result = subprocess.run(args, cwd=project_dir)

    if result.returncode == 0:
        dist_dir = os.path.join(project_dir, 'dist', 'TextStageProcessor')
        print("\n" + "=" * 60)
        print("Сборка завершена успешно!")
        print(f"Результат: {dist_dir}")
        print("=" * 60)

        # Создаём директории для входных/выходных файлов
        for d in ['input_files', 'input_files/clasterization', 'input_files/classification', 'output_files']:
            path = os.path.join(dist_dir, d)
            os.makedirs(path, exist_ok=True)
    else:
        print("\nОшибка сборки! Код возврата:", result.returncode)
        sys.exit(1)


if __name__ == '__main__':
    build()
