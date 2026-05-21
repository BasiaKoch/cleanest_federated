"""Canonical framework labels and one-stop alias normalisation.

The thesis pipeline writes a ``framework`` field into every ``test_at_best_*.json``
to record which FL runtime produced each result. Two canonical values exist:

  * ``"pure-pytorch"``    — the in-repo reference loop (``fl/server_loop.py``)
                             invoked via ``experiments/run_one.py`` and
                             back-stamped onto the headline JSONs by
                             ``scripts/backstamp_headline_provenance.py``.
  * ``"flower-simulation"`` — the Flower 1.x runtime invoked via
                              ``experiments/run_one_flower.py`` and
                              ``experiments/run_one_fednova_flower.py``.

Downstream code that filters or compares result files MUST canonicalise
the field through :func:`canonicalise_framework` rather than comparing
strings directly. This keeps any future label drift confined to the
:data:`FRAMEWORK_ALIASES` table below.

History: an earlier run of ``run_one.py`` wrote
``"pure-pytorch-reference-loop"`` rather than ``"pure-pytorch"``, which
would have caused ``analyse_system_het.py`` to reject mixed result
directories with ``ValueError``. The alias entry preserves the ability
to ingest JSONs produced before that commit landed, even though no such
file exists in the current results tree.
"""
from __future__ import annotations

from typing import Final

# The only labels code is allowed to compare against directly.
CANONICAL_FRAMEWORKS: Final[frozenset[str]] = frozenset({
    "pure-pytorch",
    "flower-simulation",
})

# Map every known-legacy label to its canonical equivalent. Keys must be
# stable strings written by older code; values must lie in CANONICAL_FRAMEWORKS.
FRAMEWORK_ALIASES: Final[dict[str, str]] = {
    "pure-pytorch-reference-loop": "pure-pytorch",
}


def canonicalise_framework(label: str) -> str:
    """Return the canonical framework label for ``label``.

    Parameters
    ----------
    label : str
        Value of the ``framework`` key from a result JSON.

    Returns
    -------
    str
        One of the strings in :data:`CANONICAL_FRAMEWORKS`.

    Raises
    ------
    ValueError
        If ``label`` is neither canonical nor a known alias. The error
        message lists the canonical set and the alias table to make
        future drift easy to fix.
    """
    if label in CANONICAL_FRAMEWORKS:
        return label
    if label in FRAMEWORK_ALIASES:
        return FRAMEWORK_ALIASES[label]
    raise ValueError(
        f"unknown framework label {label!r}; expected one of "
        f"{sorted(CANONICAL_FRAMEWORKS)} or an alias in "
        f"{sorted(FRAMEWORK_ALIASES)}"
    )
