# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../viat/run.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../viat/Icon', 'Icon'),
        ('../viat/utils', 'utils'),
        ('../viat/widgets', 'widgets')
    ],
    hiddenimports=[
        'PyQt5.QtCore', 
        'PyQt5.QtGui', 
        'PyQt5.QtWidgets',
        'cv2',
        'numpy',
        'xml.etree.ElementTree',
        'json',
        'qtawesome',
        'qtawesome.iconic_font',
        
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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