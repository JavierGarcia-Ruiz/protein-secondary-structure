"""PyTorch dataset, padding collate function and length-bucketed sampler."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ssp.data.constants import IGNORE_INDEX, Q8_TO_Q3_INDEX
from ssp.data.netsurfp import RaggedSplit

FEATURE_SETS: dict[str, slice] = {
    "onehot": slice(0, 20),
    "profile": slice(0, 50),  # one-hot + HMM profile
    "hmm_only": slice(20, 50),
}


def feature_dim(feature_set: str) -> int:
    s = FEATURE_SETS[feature_set]
    return s.stop - s.start


class SecondaryStructureDataset(Dataset):
    """Per-protein dataset returning ``(features[L, F], q8[L], q3[L])`` tensors."""

    def __init__(self, split: RaggedSplit, feature_set: str = "profile", max_len: int | None = None):
        self.split = split
        self.cols = FEATURE_SETS[feature_set]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        x, q8 = self.split.protein(i)
        if self.max_len is not None and len(q8) > self.max_len:
            x, q8 = x[: self.max_len], q8[: self.max_len]
        q8 = q8.astype(np.int64)
        q3 = np.full_like(q8, IGNORE_INDEX)
        valid = q8 != IGNORE_INDEX
        q3[valid] = Q8_TO_Q3_INDEX[q8[valid]]
        return {
            "x": torch.from_numpy(x[:, self.cols].astype(np.float32)),
            "q8": torch.from_numpy(q8),
            "q3": torch.from_numpy(q3),
            "idx": torch.tensor(i),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad to the longest protein in the batch; ``mask`` marks real residues."""
    lengths = torch.tensor([b["x"].shape[0] for b in batch])
    L, F = int(lengths.max()), batch[0]["x"].shape[1]
    B = len(batch)
    x = torch.zeros(B, L, F)
    q8 = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    q3 = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["x"].shape[0]
        x[i, :n], q8[i, :n], q3[i, :n], mask[i, :n] = b["x"], b["q8"], b["q3"], True
    return {
        "x": x, "q8": q8, "q3": q3, "mask": mask, "lengths": lengths,
        "idx": torch.stack([b["idx"] for b in batch]),
    }


class BucketBatchSampler(Sampler[list[int]]):
    """Groups proteins of similar length to minimise padding.

    Proteins are sorted by length inside random mega-buckets of
    ``bucket_factor * batch_size`` items, then batched; batch order is shuffled.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        bucket_factor: int = 50,
        seed: int = 0,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket = batch_size * bucket_factor
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self.lengths)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        batches: list[list[int]] = []
        for s in range(0, n, self.bucket):
            chunk = idx[s : s + self.bucket]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches += [chunk[i : i + self.batch_size].tolist() for i in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            order = self.rng.permutation(len(batches))
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self) -> int:
        n = len(self.lengths)
        return sum(
            -(-min(self.bucket, n - s) // self.batch_size) for s in range(0, n, self.bucket)
        )
