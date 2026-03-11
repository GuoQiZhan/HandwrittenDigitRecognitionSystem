# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# PyTorch相关的隐藏导入
hiddenimports = [
    'torch',
    'torchvision',
    'torch.nn',
    'torch.nn.functional',
    'PIL',
    'PIL.Image',
    'PIL.ImageOps',
    'numpy',
    'matplotlib',
    'matplotlib.backends.backend_tkagg',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'io',
    'threading',
    'collections',
]

# 收集PyTorch的子模块
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torchvision')

# 分析主脚本
a = Analysis(
    ['src/main.py'],
    pathex=[project_root],
    binaries=[],
    datas=[
        # 包含图标、图片等资源文件（如果有）
        # ('assets/icons', 'assets/icons'),
        # ('assets/images', 'assets/images'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不必要的包以减少体积
        'scipy',
        'pandas',
        'sklearn',
        'test',
        'unittest',
        'lib2to3',
        'pydoc_data',
        'distutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# 设置输出
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# 可执行文件配置
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HandwrittenDigitRecognizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # 使用UPX压缩（如果安装）
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 不显示控制台窗口
    icon=None,  # 图标文件路径（如果有）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 如果需要单文件输出，使用COLLECT
# 单文件模式不需要COLLECT
# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     name='HandwrittenDigitRecognizer'
# )