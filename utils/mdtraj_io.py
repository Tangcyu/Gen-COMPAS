"""Single entry point for MDTraj file and topology reads.

Trajectory readers suppress two known informational lines emitted by the
native DCD plugin. Topology loading does not need native-output filtering, but
is exposed here as well so production code does not call MDTraj readers
directly in multiple places.
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
