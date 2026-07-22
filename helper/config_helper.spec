# Build a fast-starting Linux/macOS onedir application with:
#   # On Linux this must print xft; PyInstaller bundles this Python's Tk library.
#   python -c 'import tkinter as tk; r=tk.Tk(); print(r.tk.call("tk::pkgconfig", "get", "fontsystem")); r.destroy()'
#   python -m PyInstaller --clean --noconfirm --distpath helper/bin helper/config_helper.spec

from pathlib import Path
import sys


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

# A Linux executable cannot repair a legacy no-Xft Tk after packaging.  Refuse
# that build instead of silently producing an application with jagged fonts.
if sys.platform.startswith("linux"):
    tk_libraries = [
        Path(source)
        for destination, source, _kind in a.binaries
        if Path(destination).name.startswith(("libtk8", "libtk9"))
    ]
    if not tk_libraries:
        raise SystemExit("Could not find Tk while verifying Xft support.")
    if not any(b"libXft.so" in library.read_bytes() for library in tk_libraries):
        raise SystemExit(
            "Refusing to build with a no-Xft Tk library. Use a Python/Tk "
            "installation linked to libXft and run PyInstaller again."
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
