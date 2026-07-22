from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_workflow_is_the_default_installed_command():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'gen-compas = "workflow:main"' in pyproject
    assert 'py-modules = ["workflow"]' in pyproject


def test_legacy_run_script_is_archived_not_packaged_at_top_level():
    assert not (PROJECT_ROOT / "run.py").exists()
    assert (PROJECT_ROOT / "archive" / "run.py").is_file()
