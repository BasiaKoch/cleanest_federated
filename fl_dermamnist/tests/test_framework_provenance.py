"""Guard tests for framework-label normalisation.

The thesis pipeline writes a ``framework`` field into every result JSON.
``fl_dermamnist/core/provenance.py`` is the single source of truth for the
allowed values and the alias table that maps legacy strings to canonical
ones. These tests fail loudly if either side drifts:

  * a canonical entry stops being canonical (would let unknown labels
    slip through downstream filters),
  * an alias resolves to something not in the canonical set (would let
    callers produce a non-canonical label from a "valid" alias),
  * an unknown label silently passes ``canonicalise_framework`` instead
    of raising ``ValueError`` (would defeat the no-mixing guarantee
    enforced by ``analyse_system_het.py``),
  * any of the 20 existing headline result JSONs stops canonicalising
    to ``"pure-pytorch"`` (would silently invalidate every downstream
    headline analysis).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fl_dermamnist.core.provenance import (
    CANONICAL_FRAMEWORKS,
    FRAMEWORK_ALIASES,
    canonicalise_framework,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HEADLINE_DIR = REPO_ROOT / "fl_dermamnist" / "results" / "headline"


# --------------------------------------------------------------------------- #
# 1. Canonical-set invariants                                                 #
# --------------------------------------------------------------------------- #
def test_canonical_set_is_nonempty_and_strings() -> None:
    assert CANONICAL_FRAMEWORKS, "CANONICAL_FRAMEWORKS must not be empty"
    for entry in CANONICAL_FRAMEWORKS:
        assert isinstance(entry, str) and entry, (
            f"canonical entry must be a non-empty string, got {entry!r}")


def test_every_canonical_entry_passes_canonicalise() -> None:
    """A canonical input must be returned unchanged."""
    for entry in CANONICAL_FRAMEWORKS:
        assert canonicalise_framework(entry) == entry


# --------------------------------------------------------------------------- #
# 2. Alias-table invariants                                                   #
# --------------------------------------------------------------------------- #
def test_every_alias_resolves_to_a_canonical_entry() -> None:
    """An alias key MUST map to a value already in CANONICAL_FRAMEWORKS.

    A typo on the value side would let callers produce a non-canonical
    label by passing a valid alias - exactly the failure mode the alias
    table exists to prevent.
    """
    for alias, target in FRAMEWORK_ALIASES.items():
        assert target in CANONICAL_FRAMEWORKS, (
            f"alias {alias!r} resolves to {target!r}, which is not in "
            f"CANONICAL_FRAMEWORKS={sorted(CANONICAL_FRAMEWORKS)}")
        # And canonicalise_framework must agree with the table.
        assert canonicalise_framework(alias) == target


def test_alias_keys_are_disjoint_from_canonical_set() -> None:
    """An alias key must not also be a canonical entry; the resolution
    order in canonicalise_framework would mask the conflict but the
    table itself is ambiguous if both branches name the same string."""
    overlap = set(FRAMEWORK_ALIASES) & set(CANONICAL_FRAMEWORKS)
    assert not overlap, (
        f"alias keys overlap with canonical set: {sorted(overlap)}")


# --------------------------------------------------------------------------- #
# 3. Unknown-input behaviour                                                  #
# --------------------------------------------------------------------------- #
def test_unknown_label_raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        canonicalise_framework("definitely-not-a-real-framework")
    # The error message should name both the canonical set and the
    # alias table so an operator can fix drift in one place.
    msg = str(exc.value)
    assert "pure-pytorch" in msg
    assert "flower-simulation" in msg


def test_empty_string_raises_value_error() -> None:
    with pytest.raises(ValueError):
        canonicalise_framework("")


def test_missing_marker_raises_value_error() -> None:
    """A JSON with no `framework` key falls back to the literal "<missing>"
    placeholder used by analyse_system_het.py's caller. canonicalise_framework
    must treat that as an unknown label, not silently pass it through."""
    with pytest.raises(ValueError):
        canonicalise_framework("<missing>")


# --------------------------------------------------------------------------- #
# 4. Real headline files canonicalise correctly                               #
# --------------------------------------------------------------------------- #
def test_all_20_headline_jsons_canonicalise_to_pure_pytorch() -> None:
    """Every existing headline result must resolve to "pure-pytorch".

    This is the integration check that ties the alias table to the real
    on-disk artefacts the thesis depends on. If the backstamp wave is
    ever re-run with a different label, or if any future runner writes
    into this directory with the wrong stamp, this test fails.
    """
    json_paths = sorted(HEADLINE_DIR.glob("test_at_best_*.json"))
    assert len(json_paths) == 20, (
        f"expected 20 headline JSONs in {HEADLINE_DIR}, found {len(json_paths)}")

    bad: list[tuple[str, str]] = []
    for p in json_paths:
        doc = json.loads(p.read_text())
        raw = doc.get("framework", "<missing>")
        try:
            canonical = canonicalise_framework(raw)
        except ValueError:
            bad.append((p.name, raw))
            continue
        if canonical != "pure-pytorch":
            bad.append((p.name, f"canonicalised to {canonical!r} (raw={raw!r})"))

    assert not bad, (
        f"{len(bad)} headline JSON(s) failed to canonicalise to "
        f"'pure-pytorch':\n" + "\n".join(f"  {n}: {v}" for n, v in bad))
