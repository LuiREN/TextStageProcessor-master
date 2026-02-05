# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('configuration.cfg', '.'), ('sources/russian_stop_words.txt', 'sources'), ('/usr/local/lib/python3.11/dist-packages/pymorphy2_dicts_ru', 'pymorphy2_dicts_ru')]
datas += collect_data_files('transformers')
datas += collect_data_files('safetensors')


a = Analysis(
    ['stage_text_processor.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['pymorphy2', 'pymorphy2.units', 'pymorphy2.units.by_analogy', 'pymorphy2.units.by_hyphen', 'pymorphy2.units.by_lookup', 'pymorphy2.units.by_shape', 'pymorphy2.units.unkn', 'pymorphy2_dicts_ru', 'pymorphy3', 'pymorphy3_dicts_ru', 'dawg_python', 'dawg2_python', 'sklearn.utils._typedefs', 'sklearn.utils._heap', 'sklearn.utils._sorting', 'sklearn.utils._vector_sentinel', 'sklearn.neighbors._partition_nodes', 'sklearn.tree._utils', 'gensim.models.word2vec', 'gensim.models.fasttext', 'numpy', 'scipy', 'pandas', 'matplotlib', 'matplotlib.backends.backend_qt5agg', 'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'PyQt5.sip', 'sources', 'sources.utils', 'sources.TextPreprocessing', 'sources.TextClasterization', 'sources.TextClastering', 'sources.TextClassification', 'sources.TextClassificationLib', 'sources.TextLSA', 'sources.TextDecomposeAndRuleApply', 'sources.XiSquare', 'sources.AnnotationMaker', 'sources.Word2VecNew', 'sources.Word2VecNew.DialogWord2VecMaker', 'sources.FastText', 'sources.FastText.DialogFastTextMaker', 'sources.bert', 'sources.bert.DialogBertClassifier', 'torch', 'torch.nn', 'torch.optim', 'torch.utils', 'torch.utils.data', 'transformers', 'transformers.models', 'transformers.models.bert', 'transformers.models.auto', 'safetensors', 'huggingface_hub', 'tokenizers'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torchaudio', 'torchvision', 'tensorflow', 'keras', 'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx', 'docutils', 'nvidia', 'nvidia_cublas_cu12', 'nvidia_cuda_cupti_cu12', 'nvidia_cuda_nvrtc_cu12', 'nvidia_cuda_runtime_cu12', 'nvidia_cudnn_cu12', 'nvidia_cufft_cu12', 'nvidia_curand_cu12', 'nvidia_cusolver_cu12', 'nvidia_cusparse_cu12', 'nvidia_nccl_cu12', 'nvidia_nvjitlink_cu12', 'triton'],
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
    console=True,
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
