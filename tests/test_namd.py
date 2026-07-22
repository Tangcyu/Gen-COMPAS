from pathlib import Path

from tools.namd import run_namd_workflow


def test_unbiased_only_mode_does_not_require_targets(tmp_path: Path):
    templates = tmp_path / "templates"
    templates.mkdir()
    for state in ("A", "B"):
        (templates / f"TMD.{state}.conf").write_text(
            "TMDk 1\nTMDFile output.pdb\n", encoding="utf-8"
        )
        (templates / f"Initial.{state}.conf").write_text(
            f"# initial state {state}\n", encoding="utf-8"
        )

    config = {
        "namd_path": "/bin/true",
        "template_path": str(templates),
        "output_dir": str(tmp_path / "jobs"),
        "tmd_force_constant": 10.0,
        "protocols": [
            {
                "name": state,
                "tmd_template": f"TMD.{state}.conf",
                "unbiased_template": f"Initial.{state}.conf",
            }
            for state in ("A", "B")
        ],
        "phases": {"tmd": False, "unbiased": True},
        "execution": {"device": "cpu", "parallel_jobs": 2, "dry_run": True},
    }
    results = run_namd_workflow(config)
    assert [result["job"] for result in results] == ["A", "B"]
    assert all(result["target"] is None for result in results)
    assert all(result["status"] == "dry_run" for result in results)
