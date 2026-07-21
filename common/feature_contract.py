"""Checks that RiteWeight and VCN use one identical feature representation."""

from __future__ import annotations

from typing import Any, Mapping


FEATURE_SCHEMA = "riteweight.fixed_anchor_internal_nm_radian.v1"


def validate_riteweight_vcn_featurization(
    vcn_config: Mapping[str, Any],
    riteweight_config: Mapping[str, Any],
) -> None:
    """Reject configurations that would train and evaluate VCN differently."""
    features = riteweight_config.get("features", riteweight_config)
    mode = features.get("mode")
    internal = features.get("internal_zmat", {})

    problems = []
    if mode not in ("internal_zmat", "internal_zmat_cached"):
        problems.append(
            "RiteWeight features.mode must be internal_zmat or internal_zmat_cached"
        )
    if not vcn_config.get("z_matrix", False):
        problems.append("VCN.z_matrix must be true")
    if vcn_config.get("use_all", False):
        problems.append("VCN.use_all must be false")
    if vcn_config.get("pair_distance", False):
        problems.append("VCN.pair_distance must be false")
    if vcn_config.get("periodic", False):
        problems.append(
            "VCN.periodic must be false to preserve RiteWeight's raw feature values"
        )

    rw_selection = internal.get("atomselect")
    vcn_selection = vcn_config.get("atomselect")
    if not rw_selection or rw_selection != vcn_selection:
        problems.append(
            "RiteWeight.features.internal_zmat.atomselect must exactly match "
            "VCN.atomselect"
        )
    if internal.get("atom_order") is not None:
        problems.append(
            "RiteWeight.features.internal_zmat.atom_order must be null when VCN uses atomselect"
        )
    if internal.get("max_atoms") is not None:
        problems.append(
            "RiteWeight.features.internal_zmat.max_atoms must be null for VCN consistency"
        )
    if internal.get("order", "index") != "index":
        problems.append("RiteWeight internal atom order must be index")

    if problems:
        details = "\n  - ".join(problems)
        raise ValueError(
            "RiteWeight/VCN featurization mismatch:\n  - " + details
        )
