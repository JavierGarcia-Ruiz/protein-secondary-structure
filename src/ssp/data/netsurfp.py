"""Download, parse and cache the NetSurfP-2.0 secondary-structure datasets."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ssp.data.constants import (
    DISORDER_MASK_COL,
    EVAL_MASK_COL,
    IGNORE_INDEX,
    ONEHOT_SLICE,
    PROFILE_SLICE,
    Q8_SLICE,
    SEQ_MASK_COL,
)

log = logging.getLogger(__name__)

BASE_URL = "https://services.healthtech.dtu.dk/services/NetSurfP-2.0/training_data/"
FILES: dict[str, str] = {
    "train": "Train_HHblits.npz",
    "cb513": "CB513_HHblits.npz",
    "ts115": "TS115_HHblits.npz",
    "casp12": "CASP12_HHblits.npz",
}


def download(raw_dir: Path, overwrite: bool = False) -> None:
    """Download the four HHblits ``.npz`` files with ``curl``.

    ``--ssl-no-revoke`` is passed because Windows' schannel cannot always reach
    the certificate revocation list of the DTU server; the TLS certificate is
    still validated.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname in FILES.values():
        out = raw_dir / fname
        if out.exists() and not overwrite:
            log.info("exists: %s (%.1f MB)", out.name, out.stat().st_size / 1e6)
            continue
        log.info("downloading %s", fname)
        subprocess.run(
            ["curl", "-sSL", "--ssl-no-revoke", "--retry", "3", "-o", str(out), BASE_URL + fname],
            check=True,
        )


@dataclass
class RaggedSplit:
    """Variable-length proteins stored as concatenated residue arrays.

    Attributes:
        ids: PDB chain identifiers, shape ``(N,)``.
        offsets: start index of each protein in the residue arrays, shape ``(N + 1,)``.
        features: per-residue features ``[one-hot(20) | HMM profile(30)]``, shape ``(R, 50)``.
        q8: per-residue Q8 label index or ``IGNORE_INDEX``, shape ``(R,)``.
    """

    ids: np.ndarray
    offsets: np.ndarray
    features: np.ndarray
    q8: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def protein(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.offsets[i], self.offsets[i + 1]
        return self.features[s:e], self.q8[s:e]

    def subset(self, idx: np.ndarray) -> RaggedSplit:
        idx = np.asarray(idx)
        feats, labs, offs = [], [], [0]
        for i in idx:
            f, q = self.protein(int(i))
            feats.append(f)
            labs.append(q)
            offs.append(offs[-1] + len(q))
        return RaggedSplit(
            ids=self.ids[idx],
            offsets=np.asarray(offs, dtype=np.int64),
            features=np.concatenate(feats),
            q8=np.concatenate(labs),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=self.ids, offsets=self.offsets, features=self.features, q8=self.q8)

    @classmethod
    def load(cls, path: Path) -> RaggedSplit:
        d = np.load(path, allow_pickle=False)
        return cls(ids=d["ids"], offsets=d["offsets"], features=d["features"], q8=d["q8"])


def parse_netsurfp(npz_path: Path, use_eval_mask: bool = True, chunk: int = 512) -> RaggedSplit:
    """Convert a padded NetSurfP-2.0 ``.npz`` into a :class:`RaggedSplit`.

    Residues that are disordered/unresolved (disorder mask = 0) or outside the
    evaluation mask receive ``IGNORE_INDEX`` so they never contribute to the
    loss or to metrics (the same convention as NetSurfP-2.0).
    """
    d = np.load(npz_path)
    ids = d["pdbids"].astype(str)
    data = d["data"]  # loaded lazily per chunk below
    n = data.shape[0]
    feats, labs, offs = [], [], [0]
    for s in range(0, n, chunk):
        block = data[s : s + chunk]
        for p in block:
            seq_mask = p[:, SEQ_MASK_COL] == 1
            L = int(seq_mask.sum())
            p = p[:L]
            x = np.concatenate([p[:, ONEHOT_SLICE], p[:, PROFILE_SLICE]], axis=1)
            q8 = p[:, Q8_SLICE].argmax(1).astype(np.int64)
            valid = (p[:, DISORDER_MASK_COL] == 1) & (p[:, Q8_SLICE].sum(1) > 0)
            if use_eval_mask:
                valid &= p[:, EVAL_MASK_COL] == 1
            q8[~valid] = IGNORE_INDEX
            feats.append(x.astype(np.float16))
            labs.append(q8.astype(np.int16))
            offs.append(offs[-1] + L)
    return RaggedSplit(
        ids=ids,
        offsets=np.asarray(offs, dtype=np.int64),
        features=np.concatenate(feats),
        q8=np.concatenate(labs),
    )


def train_val_split(n: int, n_val: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random protein-level split of the training set.

    The NetSurfP-2.0 training set was culled with PISCES at 25 % sequence
    identity, so a random protein-level hold-out is already non-redundant.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])
