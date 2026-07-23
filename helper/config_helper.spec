# Build a fast-starting Linux onedir application with:
#   python -m PyInstaller --clean --noconfirm --distpath helper/bin helper/config_helper.spec

from pathlib import Path


project_root = Path(SPECPATH).parent
font_search_roots = (
    Path("/usr/share/fonts/dejavu-sans-fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
)
font_datas = []
for font_name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
    font_path = next(
        (root / font_name for root in font_search_roots if (root / font_name).is_file()),
        None,
    )
    if font_path is None:
        raise SystemExit(
            f"Could not find {font_name}; install the DejaVu Sans font package "
            "before building the configuration helper."
        )
    font_datas.append((str(font_path), "fonts"))

a = Analysis(
    [str(project_root / "helper" / "config_helper.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[(str(project_root / "figures" / "scheme.png"), "figures"), *font_datas],
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
