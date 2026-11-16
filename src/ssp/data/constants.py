"""Label alphabets and the NetSurfP-2.0 ``.npz`` column layout.

The layout follows the README shipped with the NetSurfP-2.0 training data
(Klausen et al., 2019, *Proteins* 87:520-527)::

    [0:20]  amino-acid one-hot (order A C D E F G H I K L M N P Q R S T V W Y)
    [20:50] HHblits HMM profile (20 emissions + 7 transitions + 3 Neff values)
    [50]    sequence mask (1 = residue, 0 = padding)
    [51]    disorder mask (0 = disordered / unresolved, 1 = ordered)
    [52]    evaluation mask (1 = evaluate; used for CB513)
    [53:57] ASA / RSA (isolated & complexed)
    [57:65] Q8 one-hot in the order G H I B E S T C
    [65:67] phi, psi
    [67]    ASA max
"""

from __future__ import annotations

import numpy as np

AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

ONEHOT_SLICE = slice(0, 20)
PROFILE_SLICE = slice(20, 50)
SEQ_MASK_COL = 50
DISORDER_MASK_COL = 51
EVAL_MASK_COL = 52
Q8_SLICE = slice(57, 65)

#: DSSP 8-state alphabet in the column order used by NetSurfP-2.0.
Q8_LABELS: tuple[str, ...] = ("G", "H", "I", "B", "E", "S", "T", "C")
Q8_CHARS: tuple[str, ...] = Q8_LABELS
#: 3-state alphabet.
Q3_LABELS: tuple[str, ...] = ("H", "E", "C")
Q3_CHARS: tuple[str, ...] = Q3_LABELS
AA_CHARS: tuple[str, ...] = tuple(AMINO_ACIDS)

Q8_NAMES: dict[str, str] = {
    "G": "3₁₀-helix",
    "H": "α-helix",
    "I": "π-helix",
    "B": "β-bridge",
    "E": "β-strand",
    "S": "bend",
    "T": "turn",
    "C": "coil",
}

#: Standard reduction (H, G, I -> H; E, B -> E; S, T, C -> C).
Q8_TO_Q3: dict[str, str] = {
    "G": "H", "H": "H", "I": "H",
    "B": "E", "E": "E",
    "S": "C", "T": "C", "C": "C",
}

#: Vectorised index map: Q8 index -> Q3 index.
Q8_TO_Q3_INDEX: np.ndarray = np.array(
    [Q3_LABELS.index(Q8_TO_Q3[q]) for q in Q8_LABELS], dtype=np.int64
)

IGNORE_INDEX: int = -100


def q8_to_q3(q8: np.ndarray) -> np.ndarray:
    """Map an array of Q8 indices to Q3 indices, preserving ``IGNORE_INDEX``."""
    q8 = np.asarray(q8)
    out = np.full(q8.shape, IGNORE_INDEX, dtype=np.int64)
    valid = q8 != IGNORE_INDEX
    out[valid] = Q8_TO_Q3_INDEX[q8[valid]]
    return out
