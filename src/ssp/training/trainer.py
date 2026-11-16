"""Training loop, multi-task loss calculation and evaluation engine."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssp.data.constants import IGNORE_INDEX
from ssp.data.dataset import BucketBatchSampler, SecondaryStructureDataset, collate
from ssp.data.netsurfp import RaggedSplit
from ssp.evaluation.metrics import compute_metrics, compute_sov
from ssp.utils import get_logger


class Trainer:
    """Trainer for multi-task secondary structure prediction models."""

    def __init__(
        self,
        model: nn.Module,
        cfg: dict[str, Any],
        device: torch.device,
        logger=None,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.logger = logger or get_logger("ssp.trainer")
        
        train_cfg = cfg.get("training", {})
        self.lr = train_cfg.get("lr", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.epochs = train_cfg.get("epochs", 15)
        self.q3_weight = train_cfg.get("q3_loss_weight", 0.3)
        self.patience = train_cfg.get("early_stop_patience", 5)
        self.use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
        
        self.criterion_q8 = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.criterion_q3 = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-5)

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss, total_q8_loss, total_q3_loss = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in loader:
            x = batch["x"].to(self.device)
            q8 = batch["q8"].to(self.device)
            q3 = batch["q3"].to(self.device)
            mask = batch["mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                out = self.model(x, mask=mask, lengths=lengths, tags=q8.to(self.device))
                
                # Check for CRF loss or standard cross-entropy
                if "crf_loss" in out and out["crf_loss"] is not None:
                    loss_q8 = out["crf_loss"]
                else:
                    loss_q8 = self.criterion_q8(out["q8_logits"].view(-1, 8), q8.view(-1))
                    
                loss_q3 = self.criterion_q3(out["q3_logits"].view(-1, 3), q3.view(-1))
                loss = loss_q8 + self.q3_weight * loss_q3

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            total_q8_loss += loss_q8.item()
            total_q3_loss += loss_q3.item()
            n_batches += 1

        self.scheduler.step()
        return {
            "loss": total_loss / max(n_batches, 1),
            "loss_q8": total_q8_loss / max(n_batches, 1),
            "loss_q3": total_q3_loss / max(n_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        all_q8_true, all_q8_pred = [], []
        all_q3_true, all_q3_pred = [], []
        all_embeddings = []

        for batch in loader:
            x = batch["x"].to(self.device)
            q8 = batch["q8"]
            q3 = batch["q3"]
            mask = batch["mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                out = self.model(x, mask=mask, lengths=lengths, tags=q8.to(self.device))
                
            if hasattr(self.model, "decode_q8"):
                pred_paths = self.model.decode_q8(out["q8_logits"], mask)
                q8_p_flat = []
                for p in pred_paths:
                    q8_p_flat.extend(p)
                q8_pred_arr = np.array(q8_p_flat, dtype=np.int64)
                q8_true_arr = q8[mask.cpu()].numpy()
                all_q8_pred.append(q8_pred_arr)
                all_q8_true.append(q8_true_arr)

                pred_q3 = torch.argmax(out["q3_logits"], dim=-1).cpu().numpy()
                all_q3_pred.append(pred_q3[mask.cpu().numpy()])
                all_q3_true.append(q3[mask.cpu()].numpy())
            else:
                pred_q8 = torch.argmax(out["q8_logits"], dim=-1).cpu().numpy()
                pred_q3 = torch.argmax(out["q3_logits"], dim=-1).cpu().numpy()
                all_q8_pred.append(pred_q8)
                all_q8_true.append(q8.numpy())
                all_q3_pred.append(pred_q3)
                all_q3_true.append(q3.numpy())

            if len(all_embeddings) < 5:
                # Cache small subset of residue embeddings for latent UMAP
                emb = out["embeddings"].cpu().float()
                valid = mask.cpu() & (q8 != IGNORE_INDEX)
                for b_i in range(emb.shape[0]):
                    v_idx = valid[b_i]
                    if v_idx.sum() > 0:
                        all_embeddings.append({
                            "rep": emb[b_i, v_idx].numpy(),
                            "q8": q8[b_i, v_idx].numpy(),
                            "q3": q3[b_i, v_idx].numpy(),
                        })

        if hasattr(self.model, "decode_q8"):
            y8_t = np.concatenate(all_q8_true)
            y8_p = np.concatenate(all_q8_pred)
            y3_t = np.concatenate(all_q3_true)
            y3_p = np.concatenate(all_q3_pred)
            metrics = compute_metrics(y8_t, y8_p, q3_true=y3_t, q3_pred=y3_p)
        else:
            y8_t = np.concatenate([a.flatten() for a in all_q8_true])
            y8_p = np.concatenate([a.flatten() for a in all_q8_pred])
            y3_t = np.concatenate([a.flatten() for a in all_q3_true])
            y3_p = np.concatenate([a.flatten() for a in all_q3_pred])
            metrics = compute_metrics(y8_t, y8_p, q3_true=y3_t, q3_pred=y3_p)

        metrics["sample_embeddings"] = all_embeddings
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Path | None = None,
    ) -> list[dict[str, float]]:
        best_q8_acc = -1.0
        patience_left = self.patience
        history: list[dict[str, float]] = []

        self.logger.info(f"Starting training for {self.epochs} epochs on {self.device}...")
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            tr_res = self.train_epoch(train_loader)
            val_res = self.evaluate(val_loader)
            elapsed = time.time() - t0

            q8_acc = val_res["q8_acc"]
            q3_acc = val_res["q3_acc"]
            
            entry = {
                "epoch": epoch,
                "train_loss": tr_res["loss"],
                "val_q8_acc": q8_acc,
                "val_q3_acc": q3_acc,
                "val_q8_f1": val_res["q8_macro_f1"],
                "val_q3_f1": val_res["q3_macro_f1"],
                "time_sec": elapsed,
            }
            history.append(entry)
            
            self.logger.info(
                f"Epoch {epoch:02d}/{self.epochs:02d} | Loss: {tr_res['loss']:.4f} | "
                f"Val Q8: {q8_acc:.4f} | Val Q3: {q3_acc:.4f} | {elapsed:.1f}s"
            )

            if q8_acc > best_q8_acc:
                best_q8_acc = q8_acc
                patience_left = self.patience
                if save_path is not None:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Load best weights
        if save_path is not None and save_path.exists():
            self.model.load_state_dict(torch.load(save_path, map_location=self.device, weights_only=True))
            
        return history
