from pathlib import Path

import pytest
import yaml

from helper.config_helper import (
    RENDER_FONT_PATHS,
    complete_config,
    get_tk_font_backend,
    load_complete_config,
    parse_field_value,
    render_supersampled_text,
    save_complete_config,
    scaled_font_pixel_size,
    validate_workflow_config,
)


def test_render_font_lookup_prefers_packaged_regular_and_bold_fonts():
    assert RENDER_FONT_PATHS["regular"][0].name == "DejaVuSans.ttf"
    assert RENDER_FONT_PATHS["bold"][0].name == "DejaVuSans-Bold.ttf"
    assert RENDER_FONT_PATHS["regular"][0].parent.name == "fonts"


def test_supersampled_title_render_has_transparent_antialiased_edges():
    from PIL import ImageFont

    source = next(path for path in RENDER_FONT_PATHS["bold"] if path.is_file())
    font = ImageFont.truetype(str(source), 80)
    rendered = render_supersampled_text(
        "Gen-COMPAS", font, "#12344d", supersample=4
    )

    assert rendered.mode == "RGBA"
    assert rendered.width > rendered.height > 1
    alpha_values = set(rendered.getchannel("A").getdata())
    assert 0 in alpha_values
    assert len(alpha_values) > 2


def test_complete_config_expands_defaults_and_preserves_overrides():
    config = complete_config(
        {
            "VCN": {"q_variance": 0.2},
            "Custom": {"preserved": True},
        }
    )

    assert config["VCN"]["q_variance"] == 0.2
    assert config["VCN"]["n_targets"] == 20
    assert config["Workflow"]["warm_start_diffusion"] is True
    assert config["Custom"]["preserved"] is True


@pytest.mark.parametrize(
    ("text", "expected", "result"),
    [
        ("12", 1, 12),
        ("1.25", 1.0, 1.25),
        ("1e-06", 1.0, 1.0e-6),
        ("-2E+03", 1.0, -2000.0),
        ("false", True, False),
        ("[A, B]", [], ["A", "B"]),
        ("{0: 50, 1: 10}", {}, {0: 50, 1: 10}),
        ("cuda:0", "cpu", "cuda:0"),
        ("", None, None),
    ],
)
def test_parse_field_value_uses_schema_type(text, expected, result):
    assert parse_field_value(text, expected, "Test.value") == result


def test_complete_config_yaml_round_trip(tmp_path: Path):
    output = tmp_path / "complete.yaml"
    config = complete_config({"Workflow": {"root_dir": "./custom"}})

    save_complete_config(config, output)
    loaded = load_complete_config(output)

    assert loaded == config
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["VCN"]["n_targets"] == 20


def test_validation_accepts_configured_complete_workflow():
    config = complete_config(
        {
            "Workflow": {
                "initial_diffusion_data": {
                    "dcd_path": "/data/initial.dcd",
                    "topology_path": "/data/initial.psf",
                },
                "run_fel": False,
            },
            "Occupancy": {
                "topology_file": "/data/full.psf",
                "pdb_file": "/data/full.pdb",
            },
            "NAMD": {
                "namd_path": "namd3",
                "template_path": "/data/NAMD_inputs",
            },
            "RiteWeight": {
                "io": {"topology": "/data/full.psf"},
                "committor_labels": {"enabled": False},
            },
        }
    )

    assert validate_workflow_config(config) == []


def test_validation_reports_missing_inputs_and_invalid_q_variance():
    config = complete_config({"VCN": {"q_variance": 0.75}})

    errors = validate_workflow_config(config)

    assert any("initial unbiased trajectory" in error for error in errors)
    assert any("VCN.q_variance" in error for error in errors)


def test_legacy_riteweight_top_key_is_migrated():
    config = complete_config({"RiteWeight": {"io": {"top": "/data/legacy.psf"}}})

    assert config["RiteWeight"]["io"]["topology"] == "/data/legacy.psf"
    assert "top" not in config["RiteWeight"]["io"]


@pytest.mark.parametrize(
    ("points", "tk_scaling", "ui_scale", "pixel_size"),
    [
        (12, 1.0, 1.0, -12),
        (13, 4 / 3, 1.0, -17),
        (13, 4 / 3, 1.25, -22),
        (10, 2.0, 1.0, -20),
    ],
)
def test_scaled_font_pixel_size_returns_exact_device_pixels(
    points, tk_scaling, ui_scale, pixel_size
):
    assert scaled_font_pixel_size(points, tk_scaling, ui_scale) == pixel_size


def test_scaled_font_pixel_size_rejects_invalid_values():
    with pytest.raises(ValueError):
        scaled_font_pixel_size(12, 0, 1)


def test_get_tk_font_backend_reports_xft():
    class TclInterpreter:
        @staticmethod
        def call(*args):
            assert args == ("tk::pkgconfig", "get", "fontsystem")
            return "xft"

    class Root:
        tk = TclInterpreter()

    assert get_tk_font_backend(Root()) == "xft"


def test_get_tk_font_backend_handles_unsupported_query():
    class TclInterpreter:
        @staticmethod
        def call(*_args):
            raise RuntimeError("unsupported")

    class Root:
        tk = TclInterpreter()

    assert get_tk_font_backend(Root()) == "unknown"
