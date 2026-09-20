"""Scientific and execution boundaries of the diffusion noise calibration."""
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
md = pytest.importorskip("mdtraj")

from common.config import DEFAULT_CONFIG, load_config, resolve_iteration_config
from common.diffusion_autonoise import (
    DEFAULT_AUTONOISE, StructureEvaluator, _noise_bank, _distribution_figure, _save_outputs, autonoise_config,
    choose_recommendations, find_brackets, generate_mixed_noise, next_noise,
    run_autonoise, sample_reference, summarize_candidates, terminal_separability,
)
from utils.coordinate_contract import create_coordinate_contract, save_coordinate_contract
from utils.diffusion import Diffusion
from utils.model import DiffusionModel


class Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return .6 * x + t[:, None, None] * .01


def test_mixed_sampling_matches_scalar_posterior_and_preserves_rng(monkeypatch):
    diffusion = Diffusion(timesteps=7, beta_schedule="linear", device="cpu")
    requests = [(s, i) for i in range(2) for s in (0., 1., 2.5)]
    before = torch.random.get_rng_state().clone()
    model = Denoiser()
    combined = generate_mixed_noise(model, diffusion, requests, 4, "cpu", 6, 17)
    assert model.calls == 7  # Six candidates in one batch, not six denoising loops.
    assert torch.equal(before, torch.random.get_rng_state())
    singleton = generate_mixed_noise(Denoiser(), diffusion, requests, 4, "cpu", 1, 17)
    np.testing.assert_allclose(combined, singleton, atol=1e-6)
    expected = []
    for scale, sample_id in requests:
        bank = _noise_bank(17, [sample_id], 7, 4, "cpu")
        x = bank[0]
        for step in reversed(range(7)):
            monkeypatch.setattr(torch, "randn_like", lambda *a, step=step, **kw: bank[step + 1])
            t = torch.tensor([step])
            x = diffusion.p_sample(Denoiser()(x, t), x, t, scale)
        expected.append(x[0].numpy())
    np.testing.assert_allclose(combined, expected, atol=1e-6)
    assert not np.allclose(combined[0], combined[2])


def molecular_references():
    topology = md.Topology()
    points = []
    for chain_id in range(2):
        chain = topology.add_chain()
        residue = topology.add_residue("ALA", chain, resSeq=1)
        names = [("N", md.element.nitrogen), ("CA", md.element.carbon),
                 ("C", md.element.carbon), ("O", md.element.oxygen), ("CB", md.element.carbon)]
        atoms = [topology.add_atom(name, element, residue) for name, element in names]
        for i, j in [(0, 1), (1, 2), (2, 3), (1, 4)]:
            topology.add_bond(atoms[i], atoms[j])
        points.extend(np.array([[0., 0., 0.], [.145, 0., 0.], [.20, .14, 0.],
                                [.31, .15, 0.], [.15, 0., .15]]) + [chain_id * 1.5, 0., 0.])
    a = np.repeat(np.array(points, dtype=np.float32)[None], 12, axis=0)
    b = a.copy()
    b[:, 5:, 0] -= .8
    a[:, 5:, 0] += np.linspace(-.01, .01, len(a))[:, None]
    b[:, 5:, 0] += np.linspace(-.01, .01, len(b))[:, None]
    return topology, a, b


def test_geometry_rejects_broken_bonds_clashes_chirality_and_off_corridor():
    topology, a, b = molecular_references()
    cfg = deepcopy(DEFAULT_AUTONOISE)
    cfg["geometry"]["mode"] = "strict"
    cfg["reference"]["atomselect"] = "all"
    evaluator = StructureEvaluator(topology, a, b, cfg)
    middle = (a[:1] + b[:1]) / 2
    broken = middle.copy()
    broken[:, 1, 0] += .4
    clash = a[:1].copy()
    clash[:, 5:] = clash[:, :5]
    mirror = middle.copy()
    mirror[:, 4, 2] *= -1
    far = a[:1].copy()
    far[:, 5:, 1] += 10
    nan = middle.copy()
    nan[0, 0, 0] = np.nan
    result = evaluator.evaluate(np.concatenate((a[:1], b[:1], middle, broken, clash, mirror, far, nan)))
    assert list(result["labels"]) == ["A", "B", "intermediate", "invalid", "invalid", "invalid", "outlier", "invalid"]
    assert result["bond_failed"][3]
    assert result["clash_failed"][4]
    assert result["chirality_failed"][5]
    assert result["nonfinite_failed"][7]
    rows, _ = summarize_candidates({1.: result}, cfg)
    assert rows[0]["clash_failure_count"] == int(result["clash_failed"].sum())
    # A rigid transform of the complete complex cannot change distance-based labels.
    rotated = middle[:, :, [1, 2, 0]] + 12
    assert evaluator.evaluate(rotated)["labels"][0] == "intermediate"


def test_tmd_scoring_keeps_local_defects_but_excludes_outliers_and_nonfinite():
    topology, a, b = molecular_references()
    cfg = deepcopy(DEFAULT_AUTONOISE)
    evaluator = StructureEvaluator(topology, a, b, cfg)
    middle = (a[:1] + b[:1]) / 2
    defective = middle.copy()
    defective[:, 3, 0] += .4  # Broken C--O bond; CA target geometry is unchanged.
    defective[:, 4, 2] *= -1  # Wrong chirality is retained as a diagnostic.
    far = a[:1].copy()
    far[:, 5:, 0] += 10
    nan = middle.copy()
    nan[:, 3, 0] = np.nan
    result = evaluator.evaluate(np.concatenate((defective, far, nan)))
    assert result["labels"].tolist() == ["intermediate", "outlier", "invalid"]
    assert result["valid"].tolist() == [True, False, False]
    assert result["geometry_valid"].tolist() == [False, True, False]
    assert result["bond_failed"][0] and result["chirality_failed"][0]
    rows, _ = summarize_candidates({2.: result}, cfg)
    assert rows[0]["valid_fraction"] == 1 / 3
    assert not rows[0]["feasible"]
    cfg["geometry"]["mode"] = "strict"
    assert StructureEvaluator(topology, a, b, cfg).evaluate(defective)["labels"][0] == "invalid"


def test_minimal_outputs_preserve_dcd_units_and_plot_excluded_fraction(tmp_path):
    from PIL import Image
    topology, a, b = molecular_references()
    evaluator = StructureEvaluator(topology, a, b, deepcopy(DEFAULT_AUTONOISE))
    middle = (a[:1] + b[:1]) / 2
    defective = middle.copy()
    defective[:, 4, 2] *= -1
    nan = middle.copy()
    nan[:, 3, 0] = np.nan
    far = a[:1].copy()
    far[:, 5:, 0] += 10
    xyz = np.concatenate((defective, middle, nan, far))
    records = {2.75: evaluator.evaluate(xyz), .5: evaluator.evaluate(a[:1]), 5.: evaluator.evaluate(nan)}
    rows, _ = summarize_candidates(records, deepcopy(DEFAULT_AUTONOISE))
    output = _save_outputs(tmp_path, topology, {2.75: xyz, .5: a[:1], 5.: nan}, rows, [2.75], "ok", True)
    assert set(path.name for path in tmp_path.iterdir()) == {
        "noise_distribution.png", "noise_0.5.dcd", "noise_2.75.dcd"}
    assert set(output["dcd_files"]) == {"0.5", "2.75"}
    all_traj = md.load_dcd(output["dcd_files"]["2.75"], top=topology)
    np.testing.assert_allclose(all_traj.xyz, xyz[[0, 1, 3]], atol=1e-6)
    with Image.open(output["distribution_plot"]) as picture:
        picture.verify()
    figure = _distribution_figure(rows, [2.75], "ok")
    bars = figure.axes[0].containers
    assert [bar.get_height() for bar in bars[1]] == [0., .5, 0.]  # I / all samples
    assert [bar.get_height() for bar in bars[3]] == [0., .5, 1.]  # Exclusions stay visible.
    np.testing.assert_allclose(np.sum([[bar.get_height() for bar in series] for series in bars], axis=0), 1.)
    assert "2.75 *" in [label.get_text() for label in figure.axes[0].get_xticklabels()]
    figure.clear()


def test_reference_reader_is_bounded_and_converts_angstrom_to_nm(tmp_path):
    topology, a, _ = molecular_references()
    path = tmp_path / "a.dcd"
    md.Trajectory(a, topology).save_dcd(path)
    sampled, indices = sample_reference(str(path), topology, 8)
    assert len(indices) == 8 and indices[0] == 0 and indices[-1] == len(a) - 1
    np.testing.assert_allclose(sampled.xyz, a[indices], atol=1e-6)


def test_broken_references_cannot_relax_geometry_thresholds():
    topology, a, b = molecular_references()
    a[:, 3, 0] += 5
    b[:, 3, 0] += 5
    with pytest.raises(ValueError, match="Reference covalent geometry.*topology"):
        StructureEvaluator(topology, a, b, deepcopy(DEFAULT_AUTONOISE))


def test_terminal_check_has_known_chance_and_signal_limits():
    a = np.full((20, 4, 3), -1., dtype=np.float32)
    b = -a
    zero = terminal_separability(a, b, 0., .6)
    clean = terminal_separability(a, b, 1., .6)
    partial = terminal_separability(a, b, .1, .6)
    assert zero["terminal_expected_balanced_accuracy"] == .5
    assert clean["terminal_expected_balanced_accuracy"] == 1.
    assert .5 < partial["terminal_expected_balanced_accuracy"] < 1.
    # Swap the held-out labels only: fitting must not use the test block.
    a[12:], b[12:] = b[12:].copy(), a[12:].copy()
    flipped = terminal_separability(a, b, 1., .6)
    assert flipped["clean_balanced_accuracy"] == 0.


def fake_record(labels, features):
    return {"labels": np.asarray(labels), "features": np.asarray(features, dtype=float),
            "valid": np.asarray([x in ("A", "B", "intermediate") for x in labels])}


def test_scoring_deduplicates_and_never_recommends_invalid_or_unbracketed():
    cfg = deepcopy(DEFAULT_AUTONOISE)
    cfg["search"].update(verify_samples=16, min_basin_samples=8)
    records = {
        .5: fake_record(["A"] * 16, [[0, 0]] * 16),
        5.: fake_record(["B"] * 16, [[2, 2]] * 16),
        2.: fake_record(["intermediate"] * 16, [[1, 1]] * 16),
        3.: fake_record(["invalid"] * 16, [[50, 50]] * 16),
    }
    rows, clusters = summarize_candidates(records, cfg)
    middle = next(x for x in rows if x["noise_scale"] == 2.)
    assert middle["intermediate_clusters"] == 1 and middle["diversity_score"] == 1 / 16
    assert middle["score"] == 1.
    assert choose_recommendations(rows, clusters, cfg) == ("ok", [2.])
    assert choose_recommendations(rows[:-1], clusters, cfg)[0] == "unbracketed"
    invalid = [next(x for x in rows if x["noise_scale"] == 3.)]
    assert choose_recommendations(invalid, clusters, cfg) == ("no_valid_noise", [])
    cfg["search"]["verify_samples"] = 32
    assert choose_recommendations(rows, clusters, cfg)[0] == "no_verified_intermediate"


def test_primary_recommendation_prioritizes_intermediate_yield_over_diversity():
    cfg = deepcopy(DEFAULT_AUTONOISE)
    cfg["search"].update(verify_samples=16, min_basin_samples=8)
    records = {
        .5: fake_record(["A"] * 16, [[0, 0]] * 16),
        5.: fake_record(["B"] * 16, [[3, 3]] * 16),
        2.: fake_record(["intermediate"] * 16, [[1, 1]] * 16),
        3.: fake_record(["intermediate"] * 2 + ["B"] * 14, [[1.5, 1.5], [2., 2.]] + [[3., 3.]] * 14),
    }
    rows, clusters = summarize_candidates(records, cfg)
    assert choose_recommendations(rows, clusters, cfg) == ("ok", [2., 3.])


def test_search_supports_reversed_and_nonmonotone_biases():
    rows = [{"noise_scale": s, "bias": bias} for s, bias in [(1., 1), (3., -1), (5., 1)]]
    assert find_brackets(rows) == [(1., 3.), (3., 5.)]
    assert next_noise(rows, [1., 5.]) in (2., 4.)
    # A statistically uncertain candidate does not fabricate a bracket.
    assert find_brackets([{"noise_scale": 1., "bias": 0}, {"noise_scale": 5., "bias": 1}]) == []


@pytest.mark.parametrize("field,value", [("bounds", [2, 1]), ("bounds", [0, float("inf")]),
                                         ("max_batch_size", 0), ("max_samples", 2), ("refine_rounds", -1)])
def test_config_rejects_invalid_search(field, value):
    with pytest.raises(ValueError):
        autonoise_config({"Generative": {"autonoise": {"search": {field: value}}}})


def test_config_rejects_unknown_geometry_mode_and_nonboolean_dcd_flag():
    with pytest.raises(ValueError, match="geometry.mode"):
        autonoise_config({"Generative": {"autonoise": {"geometry": {"mode": "typo"}}}})
    with pytest.raises(ValueError, match="save_dcd"):
        autonoise_config({"Generative": {"autonoise": {"save_dcd": "false"}}})


def test_nested_settings_inherit_defaults_and_survive_iteration_resolution(tmp_path, monkeypatch):
    import yaml
    monkeypatch.chdir(tmp_path)
    supplied = {"Workflow": {"initial_diffusion_data": {"dcd_path": "initial.dcd", "topology_path": "top.psf"}},
                "Generative": {"autonoise": {"enabled": True, "geometry": {"mode": "strict"},
                                             "search": {"bounds": [1, 4]}}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(supplied))
    loaded = load_config(path)
    before = deepcopy(loaded)
    resolved = resolve_iteration_config(loaded, 0)
    settings = autonoise_config(resolved)
    assert loaded == before
    assert "AutoNoise" not in resolved
    assert settings["search"]["bounds"] == [1., 4.]
    assert settings["geometry"]["mode"] == "strict"
    assert settings["geometry"]["clash_distance_nm"] == DEFAULT_AUTONOISE["geometry"]["clash_distance_nm"]
    assert settings["region"] == DEFAULT_AUTONOISE["region"]
    settings["region"]["basin_margin"] = 100
    assert autonoise_config(resolved)["region"]["basin_margin"] != 100


def test_legacy_top_level_config_requires_explicit_migration():
    with pytest.raises(ValueError, match="Generative.autonoise"):
        autonoise_config({"AutoNoise": {"enabled": True}})


def make_run_config(tmp_path):
    topology, a, b = molecular_references()
    top_path = tmp_path / "top.pdb"
    md.Trajectory(a[:1], topology).save_pdb(top_path)
    topology = md.load_topology(top_path)  # Match persisted atom IDs and inferred bonds.
    for name, xyz in (("a", a), ("b", b)):
        md.Trajectory(xyz, topology).save_dcd(tmp_path / f"{name}.dcd")
    coords = torch.tensor(np.concatenate((a, b)))
    contract = create_coordinate_contract(topology=topology, reference_xyz=coords[0],
                                         alignment_atom_indices=torch.tensor([0, 1, 2]),
                                         coord_mean=coords.mean((0, 1)), coord_std=coords.std((0, 1)) + 1e-8)
    save_coordinate_contract(contract, str(tmp_path / "coordinate_contract.pt"), str(tmp_path / "ref.pdb"), topology)
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["Generative"]["device"] = "cpu"
    cfg["Generative"]["data"]["topology_path"] = str(top_path)
    model_cfg = dict(node_feature_dim=8, time_embedding_dim=8, hidden_dim=8, num_schnet_layers=1,
                     num_gat_layers=1, residue_attn_heads=2, k_neighbors=1, num_segment_layers=1, segment_distance_rbf=4)
    cfg["Generative"]["model"] = model_cfg
    cfg["Generative"]["diffusion"] = dict(timesteps=3, beta_schedule="linear")
    model = DiffusionModel(topology.n_atoms, topology, [x.name for x in topology.atoms], **model_cfg)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    cfg["Generative"]["inference"]["checkpoint"] = str(tmp_path / "model.pt")
    cfg["Generative"]["autonoise"] = deepcopy(DEFAULT_AUTONOISE)
    cfg["Generative"]["autonoise"].update(enabled=True, output_dir=str(tmp_path / "out"))
    cfg["Generative"]["autonoise"]["reference"].update(state_a=str(tmp_path / "a.dcd"), state_b=str(tmp_path / "b.dcd"))
    cfg["Generative"]["autonoise"]["search"].update(pilot_samples=2, refine_rounds=1, verify_top_k=2,
                                      verify_samples=4, max_samples=10, max_batch_size=6)
    return cfg


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_checkpoint_end_to_end_and_budget(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = make_run_config(tmp_path)
    cfg["Generative"]["device"] = device
    original = deepcopy(cfg)
    dry = run_autonoise(cfg, dry_run=True)
    assert dry["status"] == "dry_run" and not (tmp_path / "out").exists()
    report = run_autonoise(cfg)
    assert report["generated_samples"] <= 10
    assert report["status"] in ("no_valid_noise", "unbracketed", "no_verified_intermediate", "ok")
    assert cfg == original
    assert (tmp_path / "out" / "noise_distribution.png").is_file()
    assert set(path.suffix for path in (tmp_path / "out").iterdir()) <= {".png", ".dcd"}
    scores = {row["noise_scale"]: row for row in report["scores"]}
    for scale, path in report["dcd_files"].items():
        trajectory = md.load_dcd(path, top=cfg["Generative"]["data"]["topology_path"])
        assert trajectory.n_atoms == 10
        assert len(trajectory) == scores[float(scale)]["samples"]
    with pytest.raises(FileExistsError):
        run_autonoise(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_model_mixed_batch_matches_singletons():
    topology, _, _ = molecular_references()
    torch.manual_seed(42)
    model = DiffusionModel(topology.n_atoms, topology, [x.name for x in topology.atoms],
                           node_feature_dim=8, time_embedding_dim=8, hidden_dim=8,
                           num_schnet_layers=1, num_gat_layers=1, residue_attn_heads=2,
                           k_neighbors=2, num_segment_layers=1, segment_distance_rbf=4).cuda().eval()
    diffusion = Diffusion(timesteps=6, beta_schedule="linear", device="cuda")
    requests = [(s, i) for i in range(2) for s in (.5, 1., 3.)]
    mixed = generate_mixed_noise(model, diffusion, requests, topology.n_atoms, "cuda", 6, 4)
    serial = generate_mixed_noise(model, diffusion, requests, topology.n_atoms, "cuda", 1, 4)
    # Existing model uses autocast; different batch GEMMs need not be bitwise equal.
    np.testing.assert_allclose(mixed, serial, atol=3e-3, rtol=1e-2)


def test_diagnostic_only_does_not_load_or_run_model(tmp_path, monkeypatch):
    cfg = make_run_config(tmp_path)
    import common.diffusion_autonoise as module
    monkeypatch.setattr(module, "setup_model_and_diffusion", lambda *args: pytest.fail("Model must not be loaded"))
    report = run_autonoise(cfg, diagnostic_only=True)
    assert report["status"] == "diagnostic_only"
    assert "terminal_check" in report
    assert not (tmp_path / "out").exists()


def test_diagnostic_rejects_reference_atom_order_mismatch(tmp_path):
    cfg = make_run_config(tmp_path)
    for key in ("state_a", "state_b"):
        path = cfg["Generative"]["autonoise"]["reference"][key]
        trajectory = md.load(path, top=cfg["Generative"]["data"]["topology_path"])
        trajectory.xyz[:, 3, 0] += 5
        trajectory.save_dcd(path)
    with pytest.raises(ValueError, match="Reference covalent geometry.*topology"):
        run_autonoise(cfg, diagnostic_only=True)
    assert not (tmp_path / "out").exists()


def test_search_loop_finds_and_verifies_a_useful_noise(tmp_path, monkeypatch):
    cfg = make_run_config(tmp_path)
    cfg["Generative"]["autonoise"]["search"].update(pilot_samples=16, refine_rounds=2, verify_samples=32,
                                      verify_top_k=2, max_samples=112)
    cfg["Generative"]["autonoise"]["save_dcd"] = False
    import common.diffusion_autonoise as module
    contract = module.load_coordinate_contract(str(tmp_path / "coordinate_contract.pt"))
    mean = np.asarray(contract["coord_mean"]).reshape(1, 1, 3)
    std = np.asarray(contract["coord_std"]).reshape(1, 1, 3)
    _, a, b = molecular_references()

    def controlled_generator(model, diffusion, requests, *args):
        result = []
        for scale, sample_id in requests:
            if scale < 2:
                xyz = a[6]
            elif scale > 4:
                xyz = b[6]
            else:
                xyz = [a[6], b[6], (a[6] + b[6]) / 2][sample_id % 3]
            xyz = xyz.copy()
            xyz[4, 2] *= -1  # A CA TMD proposal remains usable despite a local defect.
            result.append(xyz)
        return ((np.asarray(result) - mean) / std).astype(np.float32)

    monkeypatch.setattr(module, "generate_mixed_noise", controlled_generator)
    report = run_autonoise(cfg)
    assert report["status"] == "ok"
    assert report["recommended_noise_scales"]
    assert all(2 <= s <= 4 for s in report["recommended_noise_scales"])
    assert report["generated_samples"] <= 112
    assert [path.name for path in (tmp_path / "out").iterdir()] == ["noise_distribution.png"]
    assert report["dcd_files"] == {}
    for row in report["scores"]:
        assert row["geometry_valid_fraction"] == 0.
        if row["noise_scale"] in report["recommended_noise_scales"]:
            assert row["samples"] == 32


def test_workflow_autonoise_cli_dry_run(tmp_path, capsys):
    import yaml
    import workflow
    cfg = make_run_config(tmp_path)
    cfg["Workflow"]["root_dir"] = str(tmp_path / "iterations")
    cfg["Workflow"]["initial_diffusion_data"] = {
        "dcd_path": cfg["Generative"]["autonoise"]["reference"]["state_a"],
        "topology_path": cfg["Generative"]["data"]["topology_path"],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    workflow.main(["--config", str(path), "--iteration", "0", "--run_step", "autonoise_diffusion", "--dry-run"])
    assert "pending autonoise_diffusion" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "iterations").exists()


def test_disabled_calibration_has_no_side_effects(tmp_path):
    assert run_autonoise({"Generative": {"autonoise": {"enabled": False}}})["status"] == "disabled"
    assert list(tmp_path.iterdir()) == []
