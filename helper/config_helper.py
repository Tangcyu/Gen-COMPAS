#!/usr/bin/env python3
"""Graphical editor for complete Gen-COMPAS workflow YAML files."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import (
    DEFAULT_CONFIG,
    committor_slice_bounds,
    deep_merge,
    normalize_config_aliases,
)


LOGO_PATH = PROJECT_ROOT / "figures" / "scheme.png"


REQUIRED_FIELDS = {
    "Workflow.initial_diffusion_data.dcd_path": "initial unbiased trajectory",
    "Workflow.initial_diffusion_data.topology_path": "matching initial topology",
    "Occupancy.topology_file": "reference topology containing hydrogen atoms",
    "Occupancy.pdb_file": "matching reference PDB containing hydrogen atoms",
    "NAMD.namd_path": "NAMD executable or command",
    "NAMD.template_path": "NAMD template directory",
    "RiteWeight.io.topology": "RiteWeight topology",
}

FILE_FIELDS = {
    "Workflow.initial_diffusion_data.dcd_path",
    "Workflow.initial_diffusion_data.topology_path",
    "Generative.data.dcd_path",
    "Generative.data.topology_path",
    "Generative.init_checkpoint_path",
    "Generative.inference.checkpoint",
    "VCN.topfile",
    "VCN.gendcdfile",
    "VCN.model_fn",
    "Clustering.topology",
    "Clustering.trajectory",
    "Occupancy.topology_file",
    "Occupancy.pdb_file",
    "NAMD.namd_path",
    "RiteWeight.io.topology",
    "FEL_estimate.input",
}

DIRECTORY_FIELDS = {
    "Workflow.root_dir",
    "Generative.save_dir",
    "VCN.sampling_path",
    "VCN.out_dir",
    "VCN.slice_dir",
    "Clustering.output_dir",
    "Occupancy.pdb_dir",
    "Occupancy.output_dir",
    "NAMD.template_path",
    "NAMD.output_dir",
    "NAMD.targets.path",
    "RiteWeight.io.out",
    "FEL_estimate.output_dir",
}

ENUM_FIELDS = {
    "Generative.diffusion.beta_schedule": ("cosine", "linear"),
    "NAMD.existing_job_policy": ("error", "skip"),
    "NAMD.execution.device": ("cpu", "gpu"),
    "RiteWeight.features.mode": (
        "internal_zmat",
        "internal_zmat_cached",
        "distances",
    ),
    "RiteWeight.features.cache.policy": ("write_if_missing", "force_recompute"),
    "RiteWeight.committor_labels.angle_unit": ("degree", "radian"),
}

FIELD_HELP = {
    "Workflow.initial_diffusion_data.dcd_path": "Required input for iteration 0.",
    "Workflow.initial_diffusion_data.topology_path": "Must match the initial DCD.",
    "Workflow.iteration_noise_scales": "YAML mapping, e.g. {0: 10.0, 1: 2.0}.",
    "Workflow.iteration_diffusion_epochs": "YAML mapping, e.g. {0: 50, 1: 10}.",
    "VCN.q_variance": "Half-width about q=0.5; 0.1 selects 0.4 <= q <= 0.6.",
    "VCN.n_targets": "First N committor-slice candidates; no clustering.",
    "Occupancy.pdb_dir": (
        "Workflow-managed generated target PDBs; these normally do NOT contain "
        "hydrogen atoms."
    ),
    "Occupancy.topology_file": (
        "Reference topology WITH hydrogen atoms; it must match the reference PDB."
    ),
    "Occupancy.pdb_file": (
        "Reference coordinate PDB WITH hydrogen atoms; it supplies hydrogen geometry."
    ),
    "NAMD.namd_path": "Absolute executable path or a command available on PATH.",
    "NAMD.template_path": "Directory containing TMD and unbiased configuration templates.",
    "NAMD.protocols": "YAML list of A/B (or other) protocol definitions.",
    "NAMD.execution.cpu.command": "YAML command list with workflow placeholders.",
    "NAMD.execution.gpu.command": "YAML command list with workflow placeholders.",
    "RiteWeight.committor_labels.basin_A": "Coordinates defining state A.",
    "RiteWeight.committor_labels.basin_B": "Coordinates defining state B.",
    "FEL_estimate.projections": "YAML list of one- or two-dimensional FEL projections.",
}

FIELD_LABELS = {
    "Occupancy.pdb_dir": "Generated target PDB directory (without H)",
    "Occupancy.topology_file": "Reference topology (with H)",
    "Occupancy.pdb_file": "Reference PDB coordinates (with H)",
    "RiteWeight.io.topology": "topology",
}


_MISSING = object()


def _get_path(config: Mapping[str, Any], dotted_path: str, default: Any = _MISSING):
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            if default is _MISSING:
                raise KeyError(dotted_path)
            return default
        value = value[part]
    return value


def _set_path(config: dict, path: Sequence[str], value: Any) -> None:
    current = config
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


def complete_config(user_config: Mapping[str, Any] | None = None) -> dict:
    """Return every supported setting, overridden by optional user values."""
    if user_config is None:
        return deepcopy(DEFAULT_CONFIG)
    if not isinstance(user_config, Mapping):
        raise TypeError("The top level of the configuration must be a mapping.")
    return deep_merge(DEFAULT_CONFIG, normalize_config_aliases(user_config))


def load_complete_config(path: str | Path) -> dict:
    """Load YAML and expand all omitted fields from the workflow defaults."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return complete_config(loaded)


def save_complete_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a complete configuration while preserving schema order."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(config),
            handle,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def parse_field_value(text: str, expected: Any, dotted_path: str) -> Any:
    """Convert one editor value according to its default schema type."""
    if isinstance(expected, str):
        return text
    if expected is None:
        if not text.strip():
            return None
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"{dotted_path}: invalid YAML value: {exc}") from exc
    if isinstance(expected, bool):
        normalized = text.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"{dotted_path}: expected true or false.")
        return normalized == "true"
    if isinstance(expected, float):
        try:
            return float(text.strip())
        except ValueError as exc:
            raise ValueError(f"{dotted_path}: expected a number.") from exc

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{dotted_path}: invalid YAML value: {exc}") from exc

    if isinstance(expected, list):
        if not isinstance(parsed, list):
            raise ValueError(f"{dotted_path}: expected a YAML list.")
        return parsed
    if isinstance(expected, Mapping):
        if not isinstance(parsed, Mapping):
            raise ValueError(f"{dotted_path}: expected a YAML mapping.")
        return dict(parsed)
    if isinstance(expected, int) and not isinstance(expected, bool):
        if isinstance(parsed, bool) or not isinstance(parsed, int):
            raise ValueError(f"{dotted_path}: expected an integer.")
        return parsed
    return parsed


def _looks_unset(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return True
    upper = stripped.upper()
    return "PATH/TO" in upper or "PATH_TO" in upper or "YOUR/" in upper


def validate_workflow_config(config: Mapping[str, Any]) -> list[str]:
    """Return actionable errors for inputs needed by a complete workflow."""
    errors: list[str] = []
    for dotted_path, description in REQUIRED_FIELDS.items():
        if _looks_unset(_get_path(config, dotted_path, None)):
            errors.append(f"{dotted_path}: provide the {description}.")

    try:
        committor_slice_bounds(_get_path(config, "VCN.q_variance", 0.1))
    except ValueError as exc:
        errors.append(str(exc))

    n_targets = _get_path(config, "VCN.n_targets", 20)
    if isinstance(n_targets, bool) or not isinstance(n_targets, int) or n_targets < 1:
        errors.append("VCN.n_targets must be a positive integer.")

    noise_scales = _get_path(config, "Workflow.iteration_noise_scales", {})
    if not isinstance(noise_scales, Mapping):
        errors.append("Workflow.iteration_noise_scales must be a YAML mapping.")
    diffusion_epochs = _get_path(config, "Workflow.iteration_diffusion_epochs", {})
    if not isinstance(diffusion_epochs, Mapping):
        errors.append("Workflow.iteration_diffusion_epochs must be a YAML mapping.")

    device = _get_path(config, "NAMD.execution.device", "cpu")
    if device not in {"cpu", "gpu"}:
        errors.append("NAMD.execution.device must be 'cpu' or 'gpu'.")
    protocols = _get_path(config, "NAMD.protocols", [])
    if not isinstance(protocols, list) or not protocols:
        errors.append("NAMD.protocols must contain at least one protocol.")
    else:
        for index, protocol in enumerate(protocols):
            if not isinstance(protocol, Mapping):
                errors.append(f"NAMD.protocols[{index}] must be a YAML mapping.")
                continue
            for key in ("name", "tmd_template", "unbiased_template"):
                if _looks_unset(protocol.get(key)):
                    errors.append(f"NAMD.protocols[{index}].{key} is required.")

    labels = _get_path(config, "RiteWeight.committor_labels", {})
    if isinstance(labels, Mapping) and labels.get("enabled", True):
        for key in ("basin_A", "basin_B", "basin_size"):
            if labels.get(key) is None:
                errors.append(f"RiteWeight.committor_labels.{key} is required.")

    if _get_path(config, "Workflow.run_fel", True):
        projections = _get_path(config, "FEL_estimate.projections", [])
        if not isinstance(projections, list) or not projections:
            errors.append(
                "FEL_estimate.projections needs at least one projection when "
                "Workflow.run_fel is true."
            )

    return errors


def launch_gui(
    initial_config: Mapping[str, Any],
    source_path: str | None = None,
    *,
    ui_scale: float = 1.0,
) -> None:
    """Launch the Tkinter configuration editor."""
    try:
        import tkinter as tk
        from tkinter import filedialog, font as tkfont, messagebox, ttk
        from PIL import Image, ImageTk
        import threading
    except ImportError as exc:  # pragma: no cover - depends on system Python build
        raise RuntimeError(
            "Tkinter is required for the GUI. Install your platform's Tk package "
            "or use helper/config_helper.py --output to generate YAML without a GUI."
        ) from exc

    class ScrollableFrame(ttk.Frame):
        def __init__(self, parent):
            super().__init__(parent, style="Card.TFrame")
            self.canvas = tk.Canvas(
                self,
                highlightthickness=0,
                background="#ffffff",
            )
            scrollbar = ttk.Scrollbar(
                self, orient="vertical", command=self.canvas.yview
            )
            self.inner = ttk.Frame(self.canvas, padding=18, style="Card.TFrame")
            window = self.canvas.create_window(
                (0, 0), window=self.inner, anchor="nw"
            )
            self.canvas.configure(yscrollcommand=scrollbar.set)
            self.inner.bind(
                "<Configure>",
                lambda event: self.canvas.configure(
                    scrollregion=self.canvas.bbox("all")
                ),
            )
            self.canvas.bind(
                "<Configure>",
                lambda event: self.canvas.itemconfigure(window, width=event.width),
            )
            self.canvas.bind("<Enter>", self._bind_wheel)
            self.canvas.bind("<Leave>", self._unbind_wheel)
            self.canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

        def _bind_wheel(self, _event):
            self.canvas.bind_all("<MouseWheel>", self._wheel)
            self.canvas.bind_all("<Button-4>", self._wheel_up)
            self.canvas.bind_all("<Button-5>", self._wheel_down)

        def _unbind_wheel(self, _event):
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

        def _wheel(self, event):
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

        def _wheel_up(self, _event):
            self.canvas.yview_scroll(-3, "units")

        def _wheel_down(self, _event):
            self.canvas.yview_scroll(3, "units")

    class FieldEditor:
        def __init__(self, path, expected, kind, control):
            self.path = tuple(path)
            self.expected = expected
            self.kind = kind
            self.control = control

        @property
        def dotted(self):
            return ".".join(self.path)

        def value(self):
            if self.kind == "boolean":
                return bool(self.control.get())
            if self.kind == "text":
                raw = self.control.get("1.0", "end-1c")
            else:
                raw = self.control.get()
            return parse_field_value(raw, self.expected, self.dotted)

        def set_text(self, value: str):
            if self.kind == "text":
                self.control.delete("1.0", "end")
                self.control.insert("1.0", value)
            else:
                self.control.set(value)

    class ConfigEditor:
        def __init__(self, root, config, path):
            self.root = root
            self.config = complete_config(config)
            self.current_path = Path(path).resolve() if path else None
            self.fields: dict[str, FieldEditor] = {}
            self.current_section: str | None = None
            self.nav_buttons = {}
            self.logo_image = None
            self._prepared_logo = None
            self._logo_error = None
            self.status = tk.StringVar(value="Ready")
            self.root.title("Gen-COMPAS Configuration Helper")
            self.root.geometry("1240x860")
            self.root.minsize(960, 640)
            current_scaling = float(self.root.tk.call("tk", "scaling"))
            self.root.tk.call("tk", "scaling", max(0.8, current_scaling * ui_scale))
            self._configure_styles()
            self._build_shell()
            self._rebuild_navigation()
            self._show_section(next(iter(self.config)), store_current=False)

        def _configure_styles(self):
            style = ttk.Style(self.root)
            if "clam" in style.theme_names():
                style.theme_use("clam")
            tkfont.nametofont("TkDefaultFont").configure(size=13)
            tkfont.nametofont("TkTextFont").configure(size=13)
            tkfont.nametofont("TkMenuFont").configure(size=12)
            tkfont.nametofont("TkHeadingFont").configure(size=13)
            tkfont.nametofont("TkFixedFont").configure(size=12)
            self.font_family = tkfont.nametofont("TkDefaultFont").cget("family")
            self.root.option_add("*Font", "TkDefaultFont")
            self.root.configure(background="#f3f5f7")
            style.configure("TFrame", background="#f3f5f7")
            style.configure("Card.TFrame", background="#ffffff")
            style.configure("TLabel", background="#f3f5f7", foreground="#263238")
            style.configure("Card.TLabel", background="#ffffff", foreground="#263238")
            style.configure(
                "SectionTitle.TLabel",
                background="#ffffff",
                foreground="#12344d",
                font=(self.font_family, 20, "bold"),
            )
            style.configure(
                "GroupTitle.TLabel",
                background="#ffffff",
                foreground="#176b87",
                font=(self.font_family, 13, "bold"),
                padding=(0, 10, 0, 4),
            )
            style.configure(
                "Help.TLabel",
                background="#ffffff",
                foreground="#607d8b",
                font=(self.font_family, 10),
            )
            style.configure("TButton", padding=(13, 8), font=(self.font_family, 11))
            style.configure(
                "Accent.TButton",
                background="#176b87",
                foreground="#ffffff",
                padding=(16, 8),
                font=(self.font_family, 11, "bold"),
            )
            style.map(
                "Accent.TButton",
                background=[("active", "#0f566d"), ("pressed", "#0a4458")],
            )
            style.configure("TEntry", padding=6)
            style.configure("TCombobox", padding=5)
            style.configure("TCheckbutton", background="#ffffff")

        def _build_shell(self):
            header = tk.Frame(self.root, background="#ffffff", padx=18, pady=10)
            header.pack(fill="x")
            self.logo_label = tk.Label(header, background="#ffffff")
            self.logo_label.pack(side="left", padx=(0, 16))
            title_block = tk.Frame(header, background="#ffffff")
            title_block.pack(side="left", fill="y")
            tk.Label(
                title_block,
                text="Gen-COMPAS",
                background="#ffffff",
                foreground="#12344d",
                font=(self.font_family, 23, "bold"),
            ).pack(anchor="w", pady=(12, 0))
            tk.Label(
                title_block,
                text="Workflow Configuration Helper",
                background="#ffffff",
                foreground="#607d8b",
                font=(self.font_family, 12),
            ).pack(anchor="w")
            self.root.after_idle(self._load_logo)

            toolbar = ttk.Frame(self.root, padding=(14, 10, 14, 8))
            toolbar.pack(fill="x")
            for label, command in (
                ("New defaults", self.reset_defaults),
                ("Load YAML", self.load_yaml),
                ("Validate", self.validate),
                ("Preview", self.preview),
            ):
                ttk.Button(toolbar, text=label, command=command).pack(
                    side="left", padx=(0, 6)
                )
            ttk.Button(
                toolbar,
                text="Save YAML",
                command=self.save_yaml,
                style="Accent.TButton",
            ).pack(side="left", padx=(2, 6))
            ttk.Label(
                toolbar,
                text="Fields marked * are required for a complete workflow.",
            ).pack(side="right")

            content = ttk.Frame(self.root, padding=(14, 0, 14, 10))
            content.pack(fill="both", expand=True)
            self.navigation = tk.Frame(
                content,
                background="#e8edf1",
                width=205,
                padx=8,
                pady=10,
            )
            self.navigation.pack(side="left", fill="y", padx=(0, 12))
            self.navigation.pack_propagate(False)
            self.form_host = ttk.Frame(content, style="Card.TFrame")
            self.form_host.pack(side="left", fill="both", expand=True)
            ttk.Label(
                self.root,
                textvariable=self.status,
                anchor="w",
                padding=(14, 7),
            ).pack(fill="x", side="bottom")

        def _load_logo(self):
            if not LOGO_PATH.is_file():
                self.status.set(f"Logo not found: {LOGO_PATH}")
                return

            def prepare_logo():
                try:
                    with Image.open(LOGO_PATH) as image:
                        prepared = image.convert("RGBA")
                        prepared.thumbnail((170, 125), Image.Resampling.LANCZOS)
                        self._prepared_logo = prepared.copy()
                except (OSError, ValueError) as exc:
                    self._logo_error = str(exc)

            threading.Thread(target=prepare_logo, daemon=True).start()
            self.root.after(30, self._finish_logo)

        def _finish_logo(self):
            if self._prepared_logo is not None:
                self.logo_image = ImageTk.PhotoImage(self._prepared_logo)
                self._prepared_logo = None
                self.logo_label.configure(image=self.logo_image)
                return
            if self._logo_error is not None:
                self.status.set(f"Could not display logo: {self._logo_error}")
                return
            self.root.after(30, self._finish_logo)

        def _rebuild_navigation(self):
            for child in self.navigation.winfo_children():
                child.destroy()
            self.nav_buttons.clear()
            tk.Label(
                self.navigation,
                text="SECTIONS",
                anchor="w",
                background="#e8edf1",
                foreground="#607d8b",
                font=(self.font_family, 10, "bold"),
                padx=10,
                pady=8,
            ).pack(fill="x")
            for section in self.config:
                button = tk.Button(
                    self.navigation,
                    text=str(section).replace("_", " "),
                    command=lambda name=section: self._show_section(name),
                    anchor="w",
                    relief="flat",
                    borderwidth=0,
                    highlightthickness=0,
                    padx=12,
                    pady=10,
                    background="#e8edf1",
                    activebackground="#d5e4ea",
                    foreground="#263238",
                    font=(self.font_family, 11),
                    cursor="hand2",
                )
                button.pack(fill="x", pady=1)
                self.nav_buttons[str(section)] = button

        def _store_visible_section(self):
            for editor in self.fields.values():
                _set_path(self.config, editor.path, editor.value())

        def _show_section(self, section, store_current=True):
            section = str(section)
            if section == self.current_section and self.fields:
                return
            if store_current and self.fields:
                try:
                    self._store_visible_section()
                except (TypeError, ValueError) as exc:
                    messagebox.showerror("Invalid field", str(exc))
                    self.status.set(f"Fix the current section before leaving: {exc}")
                    return

            for child in self.form_host.winfo_children():
                child.destroy()
            self.fields.clear()
            self.current_section = section
            for name, button in self.nav_buttons.items():
                selected = name == section
                button.configure(
                    background="#ffffff" if selected else "#e8edf1",
                    foreground="#176b87" if selected else "#263238",
                    font=(self.font_family, 11, "bold" if selected else "normal"),
                )

            heading = ttk.Frame(self.form_host, padding=(20, 16, 20, 8), style="Card.TFrame")
            heading.pack(fill="x")
            ttk.Label(
                heading,
                text=section.replace("_", " "),
                style="SectionTitle.TLabel",
            ).pack(anchor="w")
            ttk.Separator(self.form_host, orient="horizontal").pack(fill="x")
            scroll = ScrollableFrame(self.form_host)
            scroll.pack(fill="both", expand=True)
            values = self.config[section]
            if isinstance(values, Mapping):
                self._add_mapping(scroll.inner, values, (section,))
            else:
                self._add_field(scroll.inner, 0, section, values, (section,))
            self.status.set(f"Editing {section}")

        def _expected_value(self, path, current):
            expected = _get_path(DEFAULT_CONFIG, ".".join(path), _MISSING)
            return current if expected is _MISSING else expected

        def _add_mapping(self, parent, mapping, prefix):
            parent.columnconfigure(1, weight=1)
            row = 0
            for key, value in mapping.items():
                path = (*prefix, str(key))
                expected = self._expected_value(path, value)
                nested = (
                    isinstance(value, Mapping)
                    and bool(value)
                    and not (isinstance(expected, Mapping) and not expected)
                )
                if nested:
                    container = ttk.Frame(parent, style="Card.TFrame")
                    container.grid(
                        row=row,
                        column=0,
                        columnspan=3,
                        sticky="ew",
                        padx=3,
                        pady=(0, 5),
                    )
                    header = tk.Button(
                        container,
                    text=f"+  {str(key).replace('_', ' ')}",
                        anchor="w",
                        relief="flat",
                        borderwidth=0,
                        highlightthickness=0,
                        background="#ffffff",
                        activebackground="#eef5f7",
                        foreground="#176b87",
                        font=(self.font_family, 13, "bold"),
                        padx=2,
                        pady=8,
                        cursor="hand2",
                    )
                    header.pack(fill="x")
                    group = ttk.Frame(
                        container,
                        padding=(12, 0, 0, 8),
                        style="Card.TFrame",
                    )
                    state = {"expanded": False, "loaded": False}

                    def toggle(
                        content=group,
                        button=header,
                        child_values=value,
                        child_path=path,
                        child_key=str(key),
                        toggle_state=state,
                    ):
                        if toggle_state["expanded"]:
                            content.pack_forget()
                            button.configure(
                                text=f"+  {child_key.replace('_', ' ')}"
                            )
                            toggle_state["expanded"] = False
                            return
                        if not toggle_state["loaded"]:
                            self._add_mapping(content, child_values, child_path)
                            toggle_state["loaded"] = True
                        content.pack(fill="x")
                        button.configure(text=f"-  {child_key.replace('_', ' ')}")
                        toggle_state["expanded"] = True

                    header.configure(command=toggle)
                    row += 1
                else:
                    row += self._add_field(
                        parent, row, str(key), value, path, expected
                    )

        def _add_field(self, parent, row, key, value, path, expected=_MISSING):
            dotted = ".".join(path)
            if expected is _MISSING:
                expected = self._expected_value(path, value)
            label = FIELD_LABELS.get(dotted, key)
            label += " *" if dotted in REQUIRED_FIELDS else ""
            ttk.Label(parent, text=label, style="Card.TLabel").grid(
                row=row, column=0, sticky="nw", padx=(3, 10), pady=4
            )

            if isinstance(expected, bool):
                variable = tk.BooleanVar(value=bool(value))
                control = ttk.Checkbutton(parent, variable=variable)
                control.grid(row=row, column=1, sticky="w", pady=4)
                editor = FieldEditor(path, expected, "boolean", variable)
            elif isinstance(expected, (list, Mapping)):
                control = tk.Text(
                    parent,
                    height=6,
                    width=72,
                    wrap="none",
                    relief="solid",
                    borderwidth=1,
                    font="TkFixedFont",
                    padx=7,
                    pady=6,
                )
                rendered = yaml.safe_dump(
                    value,
                    sort_keys=False,
                    default_flow_style=isinstance(value, list)
                    and not any(isinstance(item, Mapping) for item in value),
                ).rstrip()
                control.insert("1.0", rendered)
                control.grid(row=row, column=1, sticky="ew", pady=4)
                editor = FieldEditor(path, expected, "text", control)
            else:
                variable = tk.StringVar(value="" if value is None else str(value))
                choices = ENUM_FIELDS.get(dotted)
                if choices:
                    control = ttk.Combobox(
                        parent, textvariable=variable, values=choices, state="readonly"
                    )
                else:
                    control = ttk.Entry(parent, textvariable=variable)
                control.grid(row=row, column=1, sticky="ew", pady=4)
                editor = FieldEditor(path, expected, "entry", variable)

                browse_kind = (
                    "file"
                    if dotted in FILE_FIELDS
                    else "directory" if dotted in DIRECTORY_FIELDS else None
                )
                if browse_kind:
                    ttk.Button(
                        parent,
                        text="Browse...",
                        command=lambda e=editor, kind=browse_kind: self.browse(e, kind),
                    ).grid(row=row, column=2, sticky="w", padx=(6, 3), pady=4)

            self.fields[dotted] = editor
            help_text = FIELD_HELP.get(dotted)
            if help_text:
                ttk.Label(parent, text=help_text, style="Help.TLabel").grid(
                    row=row + 1,
                    column=1,
                    columnspan=2,
                    sticky="w",
                    pady=(0, 3),
                )
                return 2
            return 1

        def browse(self, editor, kind):
            initial = editor.control.get().strip()
            initial_path = Path(initial).expanduser() if initial else Path.cwd()
            if initial_path.is_file():
                initial_dir = initial_path.parent
            elif initial_path.is_dir():
                initial_dir = initial_path
            else:
                initial_dir = Path.cwd()
            if kind == "file":
                selected = filedialog.askopenfilename(initialdir=str(initial_dir))
            else:
                selected = filedialog.askdirectory(initialdir=str(initial_dir))
            if selected:
                editor.set_text(selected)

        def collect(self):
            self._store_visible_section()
            return complete_config(self.config)

        def _collect_or_report(self):
            try:
                return self.collect()
            except (TypeError, ValueError) as exc:
                messagebox.showerror("Invalid field", str(exc))
                self.status.set(f"Invalid field: {exc}")
                return None

        def validate(self):
            config = self._collect_or_report()
            if config is None:
                return False
            errors = validate_workflow_config(config)
            if errors:
                messagebox.showerror(
                    "Configuration needs attention",
                    "\n".join(f"- {error}" for error in errors),
                )
                self.status.set(f"Validation found {len(errors)} issue(s)")
                return False
            messagebox.showinfo("Configuration valid", "All workflow inputs are set.")
            self.status.set("Configuration is valid")
            return True

        def preview(self):
            config = self._collect_or_report()
            if config is None:
                return
            text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
            window = tk.Toplevel(self.root)
            window.title("Complete YAML preview")
            window.geometry("850x700")
            editor = tk.Text(window, wrap="none")
            editor.insert("1.0", text)
            editor.configure(state="disabled")
            editor.pack(fill="both", expand=True)

        def load_yaml(self):
            selected = filedialog.askopenfilename(
                title="Load Gen-COMPAS YAML",
                filetypes=(("YAML files", "*.yaml *.yml"), ("All files", "*")),
            )
            if not selected:
                return
            try:
                self.config = load_complete_config(selected)
            except (OSError, TypeError, yaml.YAMLError) as exc:
                messagebox.showerror("Could not load YAML", str(exc))
                return
            self.current_path = Path(selected).resolve()
            self.current_section = None
            self._rebuild_navigation()
            self._show_section(next(iter(self.config)), store_current=False)
            self.status.set(f"Loaded {self.current_path}")

        def reset_defaults(self):
            if not messagebox.askyesno(
                "Reset configuration", "Replace all fields with project defaults?"
            ):
                return
            self.config = complete_config()
            self.current_path = None
            self.current_section = None
            self._rebuild_navigation()
            self._show_section(next(iter(self.config)), store_current=False)
            self.status.set("Reset to project defaults")

        def save_yaml(self):
            config = self._collect_or_report()
            if config is None:
                return
            errors = validate_workflow_config(config)
            if errors and not messagebox.askyesno(
                "Save with validation issues?",
                "The configuration has these issues:\n\n"
                + "\n".join(f"- {error}" for error in errors)
                + "\n\nSave it anyway?",
            ):
                self.status.set("Save cancelled because validation found issues")
                return
            selected = filedialog.asksaveasfilename(
                title="Save complete Gen-COMPAS YAML",
                initialdir=str(self.current_path.parent)
                if self.current_path
                else str(Path.cwd()),
                initialfile=self.current_path.name
                if self.current_path
                else "gen-compas.workflow.yaml",
                defaultextension=".yaml",
                filetypes=(("YAML files", "*.yaml"), ("All files", "*")),
            )
            if not selected:
                return
            try:
                save_complete_config(config, selected)
            except OSError as exc:
                messagebox.showerror("Could not save YAML", str(exc))
                return
            self.current_path = Path(selected).resolve()
            self.config = config
            self.status.set(f"Saved complete YAML to {self.current_path}")
            messagebox.showinfo("YAML saved", str(self.current_path))

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - requires a headless runtime
        raise RuntimeError(
            "The GUI could not connect to a display. Run it from a graphical desktop "
            "or use --output to generate a complete YAML file without a GUI."
        ) from exc
    ConfigEditor(root, initial_config, source_path)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or edit a complete Gen-COMPAS workflow YAML file."
    )
    parser.add_argument(
        "--config",
        help="Existing YAML to expand and open (or use with --output/--validate-only).",
    )
    parser.add_argument(
        "--output",
        help="Write a complete YAML non-interactively and exit.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate --config without opening the GUI.",
    )
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=1.0,
        help="Multiply automatic GUI DPI scaling (for example, 1.25).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.ui_scale <= 0:
        raise SystemExit("--ui-scale must be greater than zero.")
    if args.validate_only and not args.config:
        raise SystemExit("--validate-only requires --config.")

    try:
        config = load_complete_config(args.config) if args.config else complete_config()
    except (OSError, TypeError, yaml.YAMLError) as exc:
        raise SystemExit(f"Could not load configuration: {exc}") from exc

    if args.validate_only:
        errors = validate_workflow_config(config)
        if errors:
            raise SystemExit("Configuration is invalid:\n- " + "\n- ".join(errors))
        print("Configuration is valid.")
        return

    if args.output:
        save_complete_config(config, args.output)
        errors = validate_workflow_config(config)
        print(f"Wrote complete configuration to {Path(args.output).resolve()}")
        if errors:
            print("Complete YAML contains values that still need attention:")
            for error in errors:
                print(f"- {error}")
        return

    try:
        launch_gui(config, args.config, ui_scale=args.ui_scale)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
