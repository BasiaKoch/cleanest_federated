"""Shared, dependency-light helpers for the DermaMNIST FL package.

Currently exposes :mod:`fl_dermamnist.common.paths` - the single source of
truth for repository / results / thesis-ready directory locations, so scripts
do not rely on fragile ``Path(__file__).parents[...]`` arithmetic.
"""
