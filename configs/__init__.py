"""Packaged configuration defaults, addressed by their dotted YAML paths."""

from collections.abc import Mapping
from importlib.resources import files

import yaml


def load_defaults(name):
    """Read a fresh settings mapping without importing model dependencies."""
    resource = files(__package__).joinpath(f"{name}.yaml")
    settings = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(settings, Mapping):
        raise TypeError(f"Packaged defaults {name}.yaml must contain a mapping.")
    return dict(settings)


def load_workflow_defaults():
    """Assemble Section.yaml and Section.subsection.yaml into a fresh config.

    Files contain only values below their named path. Parent files are loaded
    first; duplicate definitions fail rather than silently shadowing a default.
    Resources are resolved inside the package, independently of the cwd.
    Full.yaml is a complete run template, not a default section.
    """
    names = [
        resource.name[:-5]
        for resource in files(__package__).iterdir()
        if resource.is_file()
        and resource.name.endswith(".yaml")
        and resource.name != "Full.yaml"
    ]
    config = {}
    for name in sorted(names, key=lambda value: (value.count("."), value)):
        parts = name.split(".")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Conflicting packaged defaults at {name}.")
        if parts[-1] in target:
            raise ValueError(f"Duplicate packaged defaults at {name}.")
        target[parts[-1]] = load_defaults(name)
    return config
