"""Unit tests for secondary structure dataset and data loading utilities."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ssp.data.constants import IGNORE_INDEX
from ssp.data.dataset import BucketBatchSampler, SecondaryStructureDataset, collate
from ssp.data.netsurfp import RaggedSplit


@pytest.fixture
def dummy_split() -> RaggedSplit:
    """Create a small synthetic RaggedSplit with 5 proteins."""
    ids = np.array(["1a2zA", "2b3yB", "3c4xD", "4d5wE", "5e6vF"])
    lengths = [5, 10, 4, 8, 2]
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total_res = offsets[-1]

    rng = np.random.RandomState(42)
    features = rng.randn(total_res, 50).astype(np.float32)
    q8 = rng.randint(0, 8, size=total_res).astype(np.int64)
    q8[2] = IGNORE_INDEX  # one non-evaluated residue

    return RaggedSplit(
        ids=ids,
        offsets=offsets,
        features=features,
        q8=q8,
    )


def test_ragged_split_len_and_subset(dummy_split: RaggedSplit) -> None:
    assert len(dummy_split) == 5
    assert len(dummy_split.lengths) == 5
    assert dummy_split.lengths[0] == 5
    assert dummy_split.lengths[1] == 10

    sub = dummy_split.subset([0, 2])
    assert len(sub) == 2
    assert list(sub.ids) == ["1a2zA", "3c4xD"]
    assert len(sub.features) == 5 + 4


def test_dataset_item(dummy_split: RaggedSplit) -> None:
    ds_prof = SecondaryStructureDataset(dummy_split, feature_set="profile")
    item0 = ds_prof[0]
    assert item0["x"].shape == (5, 50)
    assert item0["q8"].shape == (5,)
    assert len(item0["x"]) == 5
    assert item0["q8"][2] == IGNORE_INDEX

    ds_onehot = SecondaryStructureDataset(dummy_split, feature_set="onehot")
    item1 = ds_onehot[1]
    assert item1["x"].shape == (10, 20)


def test_bucket_batch_sampler_and_collate(dummy_split: RaggedSplit) -> None:
    ds = SecondaryStructureDataset(dummy_split, feature_set="profile")
    sampler = BucketBatchSampler(dummy_split.lengths, batch_size=2, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate)

    batches = list(loader)
    assert len(batches) >= 2
    b0 = batches[0]
    assert "x" in b0 and "mask" in b0 and "q8" in b0 and "q3" in b0 and "lengths" in b0
    assert b0["x"].ndim == 3  # (B, L_max, 50)
    assert b0["mask"].ndim == 2  # (B, L_max)
    assert b0["q8"].shape == b0["mask"].shape
