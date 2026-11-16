"""Per-residue accuracy, per-class F1, confusion matrices, and Segment Overlap (SOV)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from ssp.data.constants import IGNORE_INDEX, Q3_LABELS, Q8_LABELS, q8_to_q3


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute per-residue accuracy over non-ignored positions."""
    mask = y_true != IGNORE_INDEX
    if not np.any(mask):
        return 0.0
    return float(np.mean(y_true[mask] == y_pred[mask]))


def compute_metrics(
    q8_true: np.ndarray,
    q8_pred: np.ndarray,
    q3_true: np.ndarray | None = None,
    q3_pred: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """Compute comprehensive per-residue metrics for Q8 and Q3."""
    q8_true = np.asarray(q8_true).flatten()
    q8_pred = np.asarray(q8_pred).flatten()

    mask = q8_true != IGNORE_INDEX
    y8_t = q8_true[mask]
    y8_p = q8_pred[mask]

    q8_acc = float(np.mean(y8_t == y8_p)) if len(y8_t) > 0 else 0.0
    cm_q8 = confusion_matrix(y8_t, y8_p, labels=list(range(len(Q8_LABELS))))

    # Compute Q3
    if q3_true is None:
        q3_true = q8_to_q3(q8_true)
    if q3_pred is None:
        q3_pred = q8_to_q3(q8_pred)

    q3_true = np.asarray(q3_true).flatten()
    q3_pred = np.asarray(q3_pred).flatten()

    mask_q3 = q3_true != IGNORE_INDEX
    y3_t = q3_true[mask_q3]
    y3_p = q3_pred[mask_q3]

    q3_acc = float(np.mean(y3_t == y3_p)) if len(y3_t) > 0 else 0.0
    cm_q3 = confusion_matrix(y3_t, y3_p, labels=list(range(len(Q3_LABELS))))

    # Per-class report
    rep_q8 = classification_report(
        y8_t, y8_p, labels=list(range(len(Q8_LABELS))), target_names=Q8_LABELS, output_dict=True, zero_division=0
    )
    rep_q3 = classification_report(
        y3_t, y3_p, labels=list(range(len(Q3_LABELS))), target_names=Q3_LABELS, output_dict=True, zero_division=0
    )

    return {
        "q8_acc": q8_acc,
        "q3_acc": q3_acc,
        "q8_macro_f1": float(rep_q8["macro avg"]["f1-score"]),
        "q3_macro_f1": float(rep_q3["macro avg"]["f1-score"]),
        "cm_q8": cm_q8,
        "cm_q3": cm_q3,
        "rep_q8": rep_q8,
        "rep_q3": rep_q3,
        "n_residues": int(len(y8_t)),
    }


def compute_sov(y_true_seqs: list[np.ndarray], y_pred_seqs: list[np.ndarray], n_states: int = 3) -> float:
    """Compute Segment Overlap (SOV) score (Rost et al., 1994).

    Measures consistency of contiguous secondary structure segments rather than isolated point predictions.
    """
    total_len = 0
    total_sov = 0.0

    def find_segments(seq: np.ndarray, state: int) -> list[tuple[int, int]]:
        segments = []
        start = None
        for i, val in enumerate(seq):
            if val == state and start is None:
                start = i
            elif val != state and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(seq)))
        return segments

    for true_seq, pred_seq in zip(y_true_seqs, y_pred_seqs):
        mask = true_seq != IGNORE_INDEX
        t_clean = true_seq[mask]
        p_clean = pred_seq[mask]
        L = len(t_clean)
        if L == 0:
            continue

        seq_sov = 0.0
        for state in range(n_states):
            s_obs = find_segments(t_clean, state)
            s_pred = find_segments(p_clean, state)
            
            for s1_start, s1_end in s_obs:
                len_s1 = s1_end - s1_start
                for s2_start, s2_end in s_pred:
                    len_s2 = s2_end - s2_start
                    # check overlap
                    ov_start = max(s1_start, s2_start)
                    ov_end = min(s1_end, s2_end)
                    if ov_end > ov_start:
                        minov = ov_end - ov_start
                        maxov = max(s1_end, s2_end) - min(s1_start, s2_start)
                        delta = min(maxov - minov, minov, len_s1 // 2, len_s2 // 2)
                        ratio = (minov + delta) / maxov
                        seq_sov += ratio * len_s1
                        break  # matched segment

        total_sov += seq_sov
        total_len += L

    return float(total_sov / max(total_len, 1))

