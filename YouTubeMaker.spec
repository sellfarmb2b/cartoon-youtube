# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('src/templates', 'src/templates'), ('src/static', 'src/static'), ('bin/mac/ffmpeg', 'bin/mac')]
binaries = []
hiddenimports = ['requests', 'ffmpeg', 'PIL', 'PIL.Image', 'PIL.ImageOps', 'mutagen', 'mutagen.mp3', 'elevenlabs', 'elevenlabs.client', 'replicate', 'openai', 'google', 'google.generativeai', 'google.api_core', 'google.api_core.gapic_v1', 'google.api_core.retry', 'pywebview', 'appdirs', 'webbrowser', 'socket', 'threading', 'concurrent.futures', 'uuid', 'pyexpat', 'xml.parsers.expat', 'xml.parsers', 'xml']
tmp_ret = collect_all('replicate')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('google.generativeai')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['src/app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'pandas'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YouTubeMaker',
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
    name='YouTubeMaker',
)
app = BUNDLE(
    coll,
    name='YouTubeMaker.app',
    icon=None,
    bundle_identifier=None,
)
