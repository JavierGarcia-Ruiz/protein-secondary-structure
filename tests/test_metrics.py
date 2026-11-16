"""Unit tests for secondary structure evaluation metrics."""

import numpy as np
import pytest

from ssp.data.constants import IGNORE_INDEX
from ssp.evaluation.metrics import compute_metrics, compute_sov


def test_compute_metrics_perfect() -> None:
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    res = compute_metrics(y_true, y_pred)
    assert res["q8_acc"] == 1.0
    assert res["q8_macro_f1"] == 1.0
    assert res["n_residues"] == 8


def test_compute_metrics_with_ignore_index() -> None:
    y_true = np.array([0, 1, IGNORE_INDEX, 3, 4])
    y_pred = np.array([0, 1, 7, 3, 0])

    res = compute_metrics(y_true, y_pred)
    assert res["n_residues"] == 4
    # 0, 1, 3 match; 4!=0 -> 3/4 = 0.75
    assert pytest.approx(res["q8_acc"], 0.01) == 0.75


def test_compute_sov() -> None:
    # 5 identical residues -> perfect overlap SOV = 1.0
    y_true_seqs = [np.array([1, 1, 1, 1, 1])]
    y_pred_seqs = [np.array([1, 1, 1, 1, 1])]
    sov = compute_sov(y_true_seqs, y_pred_seqs, n_states=3)
    assert pytest.approx(sov, 0.01) == 1.0

    # Completely disjoint
    y_true_seqs = [np.array([1, 1, 1, 1, 1])]
    y_pred_seqs = [np.array([2, 2, 2, 2, 2])]
    sov = compute_sov(y_true_seqs, y_pred_seqs, n_states=3)
    assert sov == 0.0
