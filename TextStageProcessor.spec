# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec-файл для сборки TextStageProcessor.

Использование:
    pyinstaller TextStageProcessor.spec
"""

import os
import sys

# Определяем путь к словарям pymorphy
pymorphy_datas = []
try:
    import pymorphy2_dicts_ru
    pymorphy_datas.append(
        (os.path.dirname(pymorphy2_dicts_ru.__file__), 'pymorphy2_dicts_ru')
    )
except ImportError:
    pass

try:
    import pymorphy3_dicts_ru
    pymorphy_datas.append(
        (os.path.dirname(pymorphy3_dicts_ru.__file__), 'pymorphy3_dicts_ru')
    )
except ImportError:
    pass

a = Analysis(
    ['stage_text_processor.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('configuration.cfg', '.'),
        ('sources/russian_stop_words.txt', 'sources'),
    ] + pymorphy_datas,
    hiddenimports=[
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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
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
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TextStageProcessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TextStageProcessor',
)
