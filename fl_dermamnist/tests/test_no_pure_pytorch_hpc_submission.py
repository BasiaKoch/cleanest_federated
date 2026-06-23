"""Mechanical guard: every HPC submit script routes through a Flower runner.

The project-wide rule is that every job submitted to the HPC cluster
uses the Flower runtime. Three Flower SLURM templates exist:

  - ``slurm_template_flower.sh``        (FedAvg / FedProx)
  - ``slurm_template_system_het.sh``    (FedAvg / FedProx + stragglers)
  - ``slurm_template_fednova.sh``       (FedNova)

The pure-PyTorch reference loop entry point ``experiments/run_one.py``
is local-only and not for HPC submission; the orphan SLURM template
that previously invoked it (``slurm_template.sh``) has been deleted.

These tests fail loudly if either invariant regresses:

  1. No ``submit_*.sh`` references the deleted ``slurm_template.sh``
     (which would re-create the orphan path).
  2. No ``submit_*.sh`` invokes ``run_one.py`` directly via
     ``sbatch --wrap`` or any other channel.

Centralised baseline (``slurm_centralised.sh`` → ``run_centralised.py``)
is exempt: centralised training is the non-federated control and
Flower IS federation. The exemption is documented in the
``run_centralised.py`` docstring and in ``slurm_centralised.sh``'s
preflight block.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "infra" / "slurm"  # HPC submitters + SLURM templates


def _submit_scripts() -> list[Path]:
    return sorted(SCRIPTS_DIR.glob("submit_*.sh"))


def test_no_submit_script_references_deleted_slurm_template() -> None:
    """No submit_*.sh may reference the deleted pure-PyTorch template.

    The check uses a negative-lookahead regex so that ``slurm_template_*.sh``
    (the Flower templates) are NOT flagged — only the literal bare
    filename ``slurm_template.sh`` triggers a violation.
    """
    pat = re.compile(r"slurm_template\.sh(?![_a-zA-Z])")
    violations: list[tuple[str, int, str]] = []
    for p in _submit_scripts():
        for lineno, line in enumerate(p.read_text().splitlines(), start=1):
            if pat.search(line):
                violations.append((p.name, lineno, line.strip()))
    assert not violations, (
        "Submit script references the deleted slurm_template.sh "
        "(pure-PyTorch orphan); HPC submissions must use one of the "
        "Flower templates instead:\n"
        + "\n".join(f"  {n}:{ln}: {body}" for n, ln, body in violations)
    )


def test_no_submit_script_invokes_run_one_directly() -> None:
    """No submit_*.sh may invoke run_one.py via --wrap or any inline path.

    Allowed: indirect invocation through one of the Flower templates
    (slurm_template_flower.sh, slurm_template_system_het.sh,
    slurm_template_fednova.sh). Disallowed: any direct mention of
    ``run_one`` (the pure-PyTorch entry point), excluding the Flower
    variants ``run_one_flower`` and ``run_one_fednova_flower``.
    """
    # Match `run_one` only when it is NOT immediately followed by `_flower`
    # or `_fednova_flower`. This separates the pure-PyTorch entry point
    # from the Flower variants.
    pat = re.compile(r"\brun_one(?!_flower|_fednova_flower)\b")
    violations: list[tuple[str, int, str]] = []
    for p in _submit_scripts():
        for lineno, line in enumerate(p.read_text().splitlines(), start=1):
            if pat.search(line):
                violations.append((p.name, lineno, line.strip()))
    assert not violations, (
        "Submit script invokes the pure-PyTorch run_one entry point "
        "(LOCAL ONLY, not for HPC); use run_one_flower or "
        "run_one_fednova_flower instead:\n"
        + "\n".join(f"  {n}:{ln}: {body}" for n, ln, body in violations)
    )


def test_three_flower_templates_exist() -> None:
    """The three Flower SLURM templates are present on disk.

    Existence of these files is the precondition for every federated
    submit_*.sh to land its work. If a Flower template is renamed or
    deleted without updating every submit_*.sh, this test fails before
    the broken submission reaches sbatch.
    """
    for fname in (
        "slurm_template_flower.sh",
        "slurm_template_system_het.sh",
        "slurm_template_fednova.sh",
    ):
        p = SCRIPTS_DIR / fname
        assert p.is_file(), f"required Flower SLURM template missing: {p}"


def test_deleted_pure_pytorch_template_stays_deleted() -> None:
    """The orphan slurm_template.sh must NOT come back.

    Re-introducing it (without updating submit_*.sh in lock-step) would
    re-open the path by which pure-PyTorch JSONs could land in HPC
    output directories under the Flower-only rule.
    """
    assert not (SCRIPTS_DIR / "slurm_template.sh").exists(), (
        "slurm_template.sh has been recreated; it was deleted because "
        "it invoked the pure-PyTorch run_one.py entry point in violation "
        "of the Flower-only HPC rule. If a pure-PyTorch SLURM job is "
        "genuinely required, give it a distinct name (e.g. "
        "slurm_template_purepytorch.sh) and update this test."
    )
