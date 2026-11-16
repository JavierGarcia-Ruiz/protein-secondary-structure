"""Train and benchmark secondary structure models on NetSurfP-2.0 / CB513 / TS115 / CASP12."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ssp.data.dataset import BucketBatchSampler, SecondaryStructureDataset, collate
from ssp.data.netsurfp import RaggedSplit, parse_netsurfp, train_val_split
from ssp.models.factory import build_model
from ssp.training.trainer import Trainer
from ssp.utils import get_logger, load_config, seed_everything


def run_benchmark(config_path: str = "configs/default.yaml", quick: bool = False) -> None:
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    
    logger = get_logger("ssp.train", log_file=Path("logs/train.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    raw_dir = Path(cfg["data"]["raw_dir"])
    proc_dir = Path(cfg["data"]["processed_dir"])
    models_dir = Path("models")
    results_dir = Path("reports/results")
    proc_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse or load ragged datasets
    logger.info("Loading NetSurfP-2.0 datasets...")
    train_cache = proc_dir / "train.npz"
    if train_cache.exists():
        split_train_all = RaggedSplit.load(train_cache)
    else:
        logger.info("Parsing Train_HHblits.npz...")
        split_train_all = parse_netsurfp(raw_dir / "Train_HHblits.npz")
        split_train_all.save(train_cache)

    n_val = 500 if not quick else 100
    tr_idx, val_idx = train_val_split(len(split_train_all), n_val=n_val, seed=seed)
    split_train = split_train_all.subset(tr_idx)
    split_val = split_train_all.subset(val_idx)
    logger.info(f"Train size: {len(split_train)} proteins, Val size: {len(split_val)} proteins.")

    # Test benchmarks
    test_splits = {}
    for bench_name, fname in [("cb513", "CB513_HHblits.npz"), ("ts115", "TS115_HHblits.npz"), ("casp12", "CASP12_HHblits.npz")]:
        bench_cache = proc_dir / f"{bench_name}.npz"
        if bench_cache.exists():
            test_splits[bench_name] = RaggedSplit.load(bench_cache)
        else:
            logger.info(f"Parsing {fname}...")
            s = parse_netsurfp(raw_dir / fname)
            s.save(bench_cache)
            test_splits[bench_name] = s
        logger.info(f"Benchmark {bench_name}: {len(test_splits[bench_name])} proteins.")

    # Experiments to run
    experiments = [
        {"name": "cnn_bilstm_profile", "model": "cnn_bilstm", "feature_set": "profile", "d_in": 50},
        {"name": "cnn_bilstm_onehot", "model": "cnn_bilstm", "feature_set": "onehot", "d_in": 20},
        {"name": "cnn_profile", "model": "cnn", "feature_set": "profile", "d_in": 50},
    ]
    if not quick:
        experiments.append({"name": "bilstm_crf_profile", "model": "bilstm_crf", "feature_set": "profile", "d_in": 50})

    benchmark_records = []
    batch_size = cfg["data"]["batch_size"]
    epochs = 12 if not quick else 3

    for exp in experiments:
        exp_name = exp["name"]
        logger.info(f"\n=======================================================")
        logger.info(f"STARTING EXPERIMENT: {exp_name}")
        logger.info(f"=======================================================")
        
        # Prepare DataLoaders
        feat_set = exp["feature_set"]
        ds_tr = SecondaryStructureDataset(split_train, feature_set=feat_set, max_len=cfg["data"]["max_len"])
        ds_val = SecondaryStructureDataset(split_val, feature_set=feat_set)
        
        sampler_tr = BucketBatchSampler(split_train.lengths, batch_size=batch_size, shuffle=True, seed=seed)
        loader_tr = DataLoader(ds_tr, batch_sampler=sampler_tr, collate_fn=collate, num_workers=0)
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

        # Build Model
        cur_cfg = copy.deepcopy(cfg)
        cur_cfg["model"]["name"] = exp["model"]
        cur_cfg["model"]["d_in"] = exp["d_in"]
        cur_cfg["training"]["epochs"] = 5 if exp_name == "bilstm_crf_profile" else epochs
        
        model = build_model(cur_cfg)
        trainer = Trainer(model, cur_cfg, device=device, logger=logger)
        
        # Train or load cached checkpoint
        ckpt_path = models_dir / f"{exp_name}.pt"
        if ckpt_path.exists():
            logger.info(f"Loading existing checkpoint from {ckpt_path} (skipping retraining)...")
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        else:
            history = trainer.fit(loader_tr, loader_val, save_path=ckpt_path)

        # Evaluate across benchmarks
        for bench_name, split_bench in test_splits.items():
            logger.info(f"Evaluating {exp_name} on {bench_name.upper()}...")
            ds_te = SecondaryStructureDataset(split_bench, feature_set=feat_set)
            loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
            
            m = trainer.evaluate(loader_te)
            logger.info(f"{bench_name.upper()} | Q8 Acc: {m['q8_acc']:.4f} | Q3 Acc: {m['q3_acc']:.4f} | Q8 F1: {m['q8_macro_f1']:.4f} | Q3 F1: {m['q3_macro_f1']:.4f}")
            
            rec = {
                "experiment": exp_name,
                "model": exp["model"],
                "features": exp["feature_set"],
                "benchmark": bench_name,
                "q8_acc": m["q8_acc"],
                "q3_acc": m["q3_acc"],
                "q8_macro_f1": m["q8_macro_f1"],
                "q3_macro_f1": m["q3_macro_f1"],
                "n_residues": m["n_residues"],
            }
            benchmark_records.append(rec)
            
            # Save confusion matrix & embeddings for main model on CB513
            if exp_name == "cnn_bilstm_profile" and bench_name == "cb513":
                np.save(results_dir / "cm_q8_cb513.npy", m["cm_q8"])
                np.save(results_dir / "cm_q3_cb513.npy", m["cm_q3"])
                # Save sample embeddings for UMAP
                torch.save(m["sample_embeddings"], results_dir / "cb513_sample_embeddings.pt")

    # Save benchmark metrics
    df_bench = pd.DataFrame(benchmark_records)
    df_bench.to_csv(results_dir / "metrics_benchmarks.csv", index=False)
    with (results_dir / "metrics_benchmarks.json").open("w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)

    logger.info("All secondary structure benchmark experiments completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train secondary structure models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_benchmark(config_path=args.config, quick=args.quick)
