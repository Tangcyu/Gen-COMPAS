"""Single entry point for MDTraj file and topology reads.

Trajectory readers suppress two known informational lines emitted by the
native DCD plugin. Topology loading does not need native-output filtering, but
is exposed here as well so production code does not call MDTraj readers
directly in multiple places. ``save_psf`` writes generated topologies in a
format that MDTraj reloads without changing residue names or bonds.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
import sys
import tempfile


_FILTERED_DCD_PREFIXES = (
    b"dcdplugin) detected standard 32-bit DCD file of native endianness",
    b"dcdplugin) CHARMM format DCD file (also NAMD 2.1 and later)",
)


def _flush_native_streams() -> None:
    """Flush C stdio so redirected plugin output is captured before restore."""
    try:
        ctypes.CDLL(None).fflush(None)
    except (AttributeError, OSError):
        pass


def _without_filtered_dcd_lines(data: bytes) -> bytes:
    return b"".join(
        line
        for line in data.splitlines(keepends=True)
        if not any(line.startswith(prefix) for prefix in _FILTERED_DCD_PREFIXES)
    )


@contextmanager
def filter_dcdplugin_messages():
    """Hide two informational DCD-plugin lines while preserving other output.

    MDTraj's molfile plugin writes these messages from native code directly to
    stdout/stderr, so Python's ``warnings`` and ``redirect_stdout`` cannot
    intercept them. This context temporarily captures the two OS-level file
    descriptors, removes only the known informational lines, and replays every
    other native message unchanged.
    """
    streams = ((1, sys.stdout), (2, sys.stderr))
    saved_descriptors = {}
    captures = {}
    try:
        for descriptor, stream in streams:
            try:
                stream.flush()
            except (AttributeError, OSError):
                pass
            saved_descriptors[descriptor] = os.dup(descriptor)
            captures[descriptor] = tempfile.TemporaryFile(mode="w+b")
            os.dup2(captures[descriptor].fileno(), descriptor)
        yield
    finally:
        _flush_native_streams()
        for descriptor, stream in streams:
            try:
                stream.flush()
            except (AttributeError, OSError):
                pass
            saved = saved_descriptors.get(descriptor)
            capture = captures.get(descriptor)
            if saved is None or capture is None:
                continue
            os.dup2(saved, descriptor)
            capture.seek(0)
            retained = _without_filtered_dcd_lines(capture.read())
            if retained:
                os.write(saved, retained)
            capture.close()
            os.close(saved)


def load(*args, **kwargs):
    import mdtraj as md

    with filter_dcdplugin_messages():
        return md.load(*args, **kwargs)


def load_dcd(*args, **kwargs):
    import mdtraj as md

    with filter_dcdplugin_messages():
        return md.load_dcd(*args, **kwargs)


def load_frame(*args, **kwargs):
    import mdtraj as md

    with filter_dcdplugin_messages():
        return md.load_frame(*args, **kwargs)


def load_topology(*args, **kwargs):
    """Load an MDTraj topology through the shared MD I/O interface."""
    import mdtraj as md

    return md.load_topology(*args, **kwargs)


def iterload(*args, **kwargs):
    import mdtraj as md

    with filter_dcdplugin_messages():
        iterator = md.iterload(*args, **kwargs)
    try:
        while True:
            try:
                with filter_dcdplugin_messages():
                    chunk = next(iterator)
            except StopIteration:
                return
            yield chunk
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            with filter_dcdplugin_messages():
                close()


def _psf_segment_ids(topology) -> list[str]:
    """Return one segment ID per chain, distinct from the preceding chain."""
    segment_ids = []
    for chain in topology.chains:
        segment_id = next(
            (
                residue.segment_id.strip()
                for residue in chain.residues
                if getattr(residue, "segment_id", "").strip()
            ),
            "",
        )
        segment_id = (
            segment_id
            or str(getattr(chain, "chain_id", None) or "").strip()
            or f"C{chain.index}"
        )
        if segment_ids and segment_id == segment_ids[-1]:
            segment_id = f"{segment_id}{chain.index}"
        segment_ids.append(segment_id)
    return segment_ids


def save_psf(topology, path) -> None:
    """Write an MDTraj topology as a coordinate-free PSF.

    PDB keeps only three residue-name columns, and MDTraj writes CONECT
    records only for nonstandard residue names, so residues such as CYSP
    reload as CYS without their extra bonds. A PSF keeps full segment,
    residue, and atom names plus every bond. Charges are zero and masses come
    from elements: the file describes atom identity, not a force field.
    """
    segment_ids = _psf_segment_ids(topology)
    atom_lines = []
    for atom in topology.atoms:
        residue = atom.residue
        element = atom.element
        fields = (
            segment_ids[residue.chain.index],
            str(residue.name),
            str(atom.name),
        )
        if any(not field or any(char.isspace() for char in field) for field in fields):
            raise ValueError(
                f"Cannot write PSF atom {atom.index} with empty or whitespace "
                f"segment/residue/atom names: {fields!r}."
            )
        mass = 0.0 if element is None else float(element.mass)
        atom_type = "X" if element is None else str(element.symbol)
        atom_lines.append(
            f"{atom.index + 1:10d} {fields[0]:<8s} {residue.resSeq:<8d} "
            f"{fields[1]:<8s} {fields[2]:<8s} {atom_type:<6s} "
            f"{0.0:14.6f}{mass:14.4f}{0:12d}\n"
        )

    bond_indices = [
        index + 1
        for bond in topology.bonds
        for index in (bond[0].index, bond[1].index)
    ]
    bond_lines = [
        "".join(f"{index:10d}" for index in bond_indices[start:start + 8]) + "\n"
        for start in range(0, len(bond_indices), 8)
    ]

    # The NAMD flag tells VMD and MDAnalysis to split fields on whitespace.
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("PSF EXT NAMD\n\n")
        handle.write(f"{1:10d} !NTITLE\n")
        handle.write(" REMARKS Gen-COMPAS topology: names and bonds only\n\n")
        handle.write(f"{topology.n_atoms:10d} !NATOM\n")
        handle.writelines(atom_lines)
        handle.write("\n")
        handle.write(f"{len(bond_indices) // 2:10d} !NBOND: bonds\n")
        handle.writelines(bond_lines)
        handle.write("\n")
        for section in ("NTHETA: angles", "NPHI: dihedrals", "NIMPHI: impropers"):
            handle.write(f"{0:10d} !{section}\n\n")
