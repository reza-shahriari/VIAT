# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

build_with_ai = os.environ.get('VIAT_BUILD_WITH_AI', '0') == '1'

a = Analysis(
    ['../viat/run.py'],
    pathex=['../viat'],
    binaries=[],
    datas=[
        ('../viat/Icon', 'Icon'),
    ],
    hiddenimports=[
        'PyQt5.QtCore', 
        'PyQt5.QtGui', 
        'PyQt5.QtWidgets',
        'cv2',
        'numpy',
        'xml.etree.ElementTree',
        'xml',
        'xml.dom',
        'json',
        'qtawesome',
        'qtawesome.iconic_font',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['numpy._core'] if build_with_ai else [
        'numpy._core',
        'torch',
        'torchvision',
        'transformers',
        'ultralytics',
        'timm',
        'einops',
        'decord',
        'lmdb',
        'peft',
        'accelerate',
        'bitsandbytes',
        'safetensors',
        'huggingface_hub',
        'sentencepiece',
        'regex',
        'tokenizers',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VIAT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../viat/Icon/Icon.ico' 
)