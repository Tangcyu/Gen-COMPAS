# Build a fast-starting Linux/macOS onedir application with:
#   python -m PyInstaller --clean --noconfirm --distpath helper/bin helper/config_helper.spec

from pathlib import Path


project_root = Path(SPECPATH).parent

a = Analysis(
    [str(project_root / "helper" / "config_helper.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[(str(project_root / "figures" / "scheme.png"), "figures")],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["numpy", "pandas", "torch", "matplotlib", "scipy", "sklearn"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="gen-compas-config-helper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

app = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="gen-compas-config-helper",
)
