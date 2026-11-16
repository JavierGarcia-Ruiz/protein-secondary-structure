"""Generate publication-quality figures for the Protein Secondary Structure project.

Outputs:
    reports/figures/01_dataset_overview.png
    reports/figures/02_model_benchmark_comparison.png
    reports/figures/03_confusion_matrix_q8_cb513.png
    reports/figures/04_per_residue_prediction_track.png
    reports/figures/05_accuracy_by_length_and_position.png
    reports/figures/06_latent_space_umap.png
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import torch
import umap

from ssp.data.constants import Q8_CHARS, Q3_CHARS, AA_CHARS, IGNORE_INDEX, q8_to_q3
from ssp.data.netsurfp import RaggedSplit
from ssp.models.factory import build_model
from ssp.utils import load_config
from ssp.visualization.plotting import Q8_COLORS, Q3_COLORS, OKABE_ITO, set_style, save, panel_label

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")
PROC_DIR = Path("data/processed")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

set_style()


def fig01_dataset_overview() -> None:
    """Figure 1: Dataset statistics, sequence length distributions, and Q8/Q3 class frequencies."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Load splits
    splits = {}
    for name in ["train", "cb513", "ts115", "casp12"]:
        p = PROC_DIR / f"{name}.npz"
        if p.exists():
            splits[name] = RaggedSplit.load(p)

    if not splits:
        print("Skipping Fig 1: Processed splits not found.")
        plt.close(fig)
        return

    # Panel A: Sequence length distributions
    ax = axes[0]
    for idx, (name, split) in enumerate(splits.items()):
        lengths = split.lengths
        sns.kdeplot(lengths, ax=ax, label=f"{name.upper()} (N={len(split):,})", color=OKABE_ITO[idx % len(OKABE_ITO)], lw=2)
    ax.set_title("Sequence Length Distribution")
    ax.set_xlabel("Protein Length (amino acids)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1000)
    ax.legend(frameon=False)
    panel_label(ax, "A")

    # Panel B: Q8 Class balance in Train vs CB513
    ax = axes[1]
    v_tr = splits["train"].q8[splits["train"].q8 != IGNORE_INDEX]
    q8_counts_train = np.bincount(v_tr, minlength=8)
    q8_freq_train = q8_counts_train / max(q8_counts_train.sum(), 1) * 100

    v_cb = splits["cb513"].q8[splits["cb513"].q8 != IGNORE_INDEX]
    q8_counts_cb513 = np.bincount(v_cb, minlength=8)
    q8_freq_cb513 = q8_counts_cb513 / max(q8_counts_cb513.sum(), 1) * 100

    x = np.arange(8)
    w = 0.38
    ax.bar(x - w / 2, q8_freq_train, width=w, label="Train (N=10.8k)", color="#0072B2", alpha=0.85)
    ax.bar(x + w / 2, q8_freq_cb513, width=w, label="CB513 (N=513)", color="#D55E00", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(Q8_CHARS)
    ax.set_title("DSSP 8-State Residue Distribution")
    ax.set_xlabel("DSSP Q8 State")
    ax.set_ylabel("Residue Frequency (%)")
    ax.legend(frameon=False)
    panel_label(ax, "B")

    # Panel C: Q3 Distribution
    ax = axes[2]
    q3_tr = q8_to_q3(v_tr)
    q3_counts_train = np.bincount(q3_tr, minlength=3)
    q3_freq_train = q3_counts_train / max(q3_counts_train.sum(), 1) * 100

    q3_cb = q8_to_q3(v_cb)
    q3_counts_cb513 = np.bincount(q3_cb, minlength=3)
    q3_freq_cb513 = q3_counts_cb513 / max(q3_counts_cb513.sum(), 1) * 100

    x3 = np.arange(3)
    ax.bar(x3 - w / 2, q3_freq_train, width=w, label="Train", color="#0072B2", alpha=0.85)
    ax.bar(x3 + w / 2, q3_freq_cb513, width=w, label="CB513", color="#D55E00", alpha=0.85)
    ax.set_xticks(x3)
    ax.set_xticklabels(["H (Helix)", "E (Strand)", "C (Coil)"])
    ax.set_title("3-State Reduced Distribution")
    ax.set_xlabel("Secondary Structure Class")
    ax.set_ylabel("Residue Frequency (%)")
    ax.legend(frameon=False)
    panel_label(ax, "C")

    fig.tight_layout()
    save(fig, FIGURES_DIR / "01_dataset_overview.png")
    print("Saved reports/figures/01_dataset_overview.png")


def fig02_model_benchmark_comparison() -> None:
    """Figure 2: Benchmark comparison across CB513, TS115, and CASP12."""
    csv_path = RESULTS_DIR / "metrics_benchmarks.csv"
    if not csv_path.exists():
        print("Skipping Fig 2: metrics_benchmarks.csv not found.")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    display_names = {
        "cnn_bilstm_profile": "CNN-BiLSTM (Profile)",
        "cnn_bilstm_onehot": "CNN-BiLSTM (One-hot)",
        "cnn_profile": "CNN Baseline (Profile)",
        "bilstm_crf_profile": "BiLSTM-CRF (Profile)",
    }
    df["model_label"] = df["experiment"].map(lambda x: display_names.get(x, x))
    df["benchmark_label"] = df["benchmark"].str.upper()

    # Panel A: Q8 Accuracy
    ax = axes[0]
    sns.barplot(data=df, x="benchmark_label", y="q8_acc", hue="model_label", ax=ax, palette=OKABE_ITO)
    ax.set_title("8-State Secondary Structure Accuracy (Q8)")
    ax.set_xlabel("Benchmark Dataset")
    ax.set_ylabel("Q8 Accuracy")
    ax.set_ylim(0.45, 0.85)
    ax.legend(title="", frameon=False, loc="lower left")
    for p in ax.patches:
        h = p.get_height()
        if h > 0.1:
            ax.annotate(f"{h * 100:.1f}%", (p.get_x() + p.get_width() / 2.0, h),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    panel_label(ax, "A")

    # Panel B: Q3 Accuracy
    ax = axes[1]
    sns.barplot(data=df, x="benchmark_label", y="q3_acc", hue="model_label", ax=ax, palette=OKABE_ITO)
    ax.set_title("3-State Reduced Accuracy (Q3)")
    ax.set_xlabel("Benchmark Dataset")
    ax.set_ylabel("Q3 Accuracy")
    ax.set_ylim(0.60, 0.95)
    ax.legend(title="", frameon=False, loc="lower left")
    for p in ax.patches:
        h = p.get_height()
        if h > 0.1:
            ax.annotate(f"{h * 100:.1f}%", (p.get_x() + p.get_width() / 2.0, h),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    panel_label(ax, "B")

    fig.tight_layout()
    save(fig, FIGURES_DIR / "02_model_benchmark_comparison.png")
    print("Saved reports/figures/02_model_benchmark_comparison.png")


def fig03_confusion_matrix_q8_cb513() -> None:
    """Figure 3: Row-normalized 8x8 confusion matrix on CB513."""
    cm_path = RESULTS_DIR / "cm_q8_cb513.npy"
    if not cm_path.exists():
        print("Skipping Fig 3: cm_q8_cb513.npy not found.")
        return

    cm = np.load(cm_path)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=Q8_CHARS,
        yticklabels=Q8_CHARS,
        cbar_kws={"label": "Recall (%)"},
        ax=ax,
        linewidths=0.5,
        linecolor="#EEEEEE"
    )
    ax.set_title("CB513 Secondary Structure Confusion Matrix (Q8)\nRow-Normalized Recall (%)", pad=12)
    ax.set_xlabel("Predicted DSSP State", labelpad=8)
    ax.set_ylabel("Ground Truth DSSP State", labelpad=8)

    labels_desc = [
        "G: 3_10-helix",
        "H: Alpha-helix",
        "I: Pi-helix",
        "B: Beta-bridge",
        "E: Beta-strand",
        "S: Bend",
        "T: Turn",
        "C: Coil",
    ]
    ax.text(1.15, 0.5, "\n".join(labels_desc), transform=ax.transAxes, va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CCCCCC"))

    fig.tight_layout()
    save(fig, FIGURES_DIR / "03_confusion_matrix_q8_cb513.png")
    print("Saved reports/figures/03_confusion_matrix_q8_cb513.png")


def fig04_per_residue_prediction_track() -> None:
    """Figure 4: Per-residue secondary structure prediction track for a representative CB513 protein."""
    cb513_path = PROC_DIR / "cb513.npz"
    ckpt_path = Path("models/cnn_bilstm_profile.pt")
    if not (cb513_path.exists() and ckpt_path.exists()):
        print("Skipping Fig 4: CB513 or model checkpoint not found.")
        return

    split = RaggedSplit.load(cb513_path)
    cfg = load_config("configs/default.yaml")
    cfg["model"]["name"] = "cnn_bilstm"
    cfg["model"]["d_in"] = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # Pick a protein with length ~80-120 residues
    pick_idx = 0
    for i in range(len(split)):
        L_i = split.lengths[i]
        if 80 <= L_i <= 120:
            pick_idx = i
            break

    feats, true_q8 = split.protein(pick_idx)
    true_q3 = q8_to_q3(true_q8)
    L = len(true_q8)
    aa_indices = feats[:, :20].argmax(axis=-1)
    seq = "".join([AA_CHARS[idx] for idx in aa_indices])

    x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
    mask = torch.ones(1, L, dtype=torch.bool, device=device)
    lengths = torch.tensor([L], device=device)

    with torch.no_grad():
        out = model(x, mask=mask, lengths=lengths)
        probs_q8 = torch.softmax(out["q8_logits"], dim=-1)[0].cpu().numpy()
        pred_q8 = np.argmax(probs_q8, axis=-1)
        probs_q3 = torch.softmax(out["q3_logits"], dim=-1)[0].cpu().numpy()
        pred_q3 = np.argmax(probs_q3, axis=-1)

    fig, axes = plt.subplots(4, 1, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 3]})

    # Track 1: Ground Truth Q8
    ax = axes[0]
    for pos in range(L):
        t_label = true_q8[pos]
        c = Q8_COLORS[Q8_CHARS[t_label]] if t_label != IGNORE_INDEX else "#CCCCCC"
        ax.axvspan(pos - 0.5, pos + 0.5, color=c, alpha=0.9)
    ax.set_yticks([])
    ax.set_ylabel("True Q8", rotation=0, labelpad=30, va="center", fontweight="bold")
    ax.set_xlim(-0.5, L - 0.5)
    panel_label(ax, "A")

    # Track 2: Predicted Q8
    ax = axes[1]
    for pos in range(L):
        c = Q8_COLORS[Q8_CHARS[pred_q8[pos]]]
        ax.axvspan(pos - 0.5, pos + 0.5, color=c, alpha=0.9)
    ax.set_yticks([])
    ax.set_ylabel("Pred Q8", rotation=0, labelpad=30, va="center", fontweight="bold")
    panel_label(ax, "B")

    # Track 3: Agreement indicator
    ax = axes[2]
    valid_mask = true_q8 != IGNORE_INDEX
    match = (true_q8 == pred_q8)
    for pos in range(L):
        if not valid_mask[pos]:
            c = "#CCCCCC"
        else:
            c = "#009E73" if match[pos] else "#D55E00"
        ax.axvspan(pos - 0.5, pos + 0.5, color=c, alpha=0.9)
    ax.set_yticks([])
    acc_clean = match[valid_mask].mean() if valid_mask.sum() > 0 else 0.0
    ax.set_ylabel(f"Match ({acc_clean*100:.1f}%)", rotation=0, labelpad=35, va="center", fontweight="bold")
    panel_label(ax, "C")

    # Track 4: Q3 Class Posterior Probabilities
    ax = axes[3]
    pos_arr = np.arange(L)
    ax.plot(pos_arr, probs_q3[:, 0], label="Helix (H)", color="#0072B2", lw=2)
    ax.plot(pos_arr, probs_q3[:, 1], label="Strand (E)", color="#E69F00", lw=2)
    ax.plot(pos_arr, probs_q3[:, 2], label="Coil (C)", color="#999999", lw=1.5, linestyle="--")
    ax.set_ylabel("Posterior Probability")
    ax.set_xlabel("Residue Position")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", frameon=True)
    panel_label(ax, "D")

    if L <= 120:
        ax.set_xticks(pos_arr)
        ax.set_xticklabels([f"{aa}\n{i+1}" if (i + 1) % 5 == 0 or i == 0 else aa for i, aa in enumerate(seq)], fontsize=7)

    fig.suptitle(f"Single-Protein Secondary Structure Prediction Track (CB513 sample '{split.ids[pick_idx]}', L={L})", y=0.98, fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES_DIR / "04_per_residue_prediction_track.png")
    print("Saved reports/figures/04_per_residue_prediction_track.png")


def fig05_accuracy_by_length_and_position() -> None:
    """Figure 5: Performance stratified by sequence length and relative residue position."""
    cb513_path = PROC_DIR / "cb513.npz"
    ckpt_path = Path("models/cnn_bilstm_profile.pt")
    if not (cb513_path.exists() and ckpt_path.exists()):
        print("Skipping Fig 5: CB513 or model checkpoint not found.")
        return

    split = RaggedSplit.load(cb513_path)
    cfg = load_config("configs/default.yaml")
    cfg["model"]["name"] = "cnn_bilstm"
    cfg["model"]["d_in"] = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    length_records = []
    pos_records = []

    for i in range(len(split)):
        feats, true_q8 = split.protein(i)
        s_len = len(true_q8)
        x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
        mask = torch.ones(1, s_len, dtype=torch.bool, device=device)
        lengths = torch.tensor([s_len], device=device)

        with torch.no_grad():
            out = model(x, mask=mask, lengths=lengths)
            pred_q8 = torch.argmax(out["q8_logits"], dim=-1)[0].cpu().numpy()

        valid = true_q8 != IGNORE_INDEX
        if valid.sum() > 0:
            correct = (true_q8[valid] == pred_q8[valid])
            length_records.append({"length": s_len, "q8_acc": float(correct.mean())})

            rel_pos = np.linspace(0.0, 1.0, s_len)[valid]
            for rp, c in zip(rel_pos, correct):
                pos_records.append({"rel_pos": rp, "correct": int(c)})

    df_len = pd.DataFrame(length_records)
    df_pos = pd.DataFrame(pos_records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Accuracy vs Length Bins
    ax = axes[0]
    bins = [0, 100, 200, 300, 500, 1000]
    labels = ["<100", "100-200", "200-300", "300-500", "500+"]
    df_len["len_bin"] = pd.cut(df_len["length"], bins=bins, labels=labels)
    sns.boxplot(data=df_len, x="len_bin", y="q8_acc", ax=ax, palette="Blues", showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "#D55E00", "markeredgecolor": "black", "markersize": 7})
    ax.set_title("Q8 Accuracy vs Sequence Length")
    ax.set_xlabel("Sequence Length Bin (amino acids)")
    ax.set_ylabel("Q8 Accuracy")
    ax.set_ylim(0.4, 1.0)
    panel_label(ax, "A")

    # Panel B: Accuracy across Relative Residue Position
    ax = axes[1]
    df_pos["pos_bin"] = pd.cut(df_pos["rel_pos"], bins=np.linspace(0, 1, 11), labels=[f"{int(i*10)}%" for i in range(1, 11)])
    pos_stats = df_pos.groupby("pos_bin", observed=True)["correct"].agg(["mean", "count"]).reset_index()
    ax.plot(range(len(pos_stats)), pos_stats["mean"], marker="o", color="#0072B2", lw=2.5, markersize=8)
    ax.set_xticks(range(len(pos_stats)))
    ax.set_xticklabels(pos_stats["pos_bin"])
    ax.set_title("Q8 Accuracy Across Relative Chain Position")
    ax.set_xlabel("Relative Chain Position (N-term -> C-term)")
    ax.set_ylabel("Mean Accuracy")
    ax.set_ylim(0.55, 0.85)
    ax.axvspan(-0.5, 0.5, color="#D55E00", alpha=0.15, label="N-terminal loop flexibility")
    ax.axvspan(8.5, 9.5, color="#D55E00", alpha=0.15, label="C-terminal loop flexibility")
    ax.legend(frameon=False, loc="lower center")
    panel_label(ax, "B")

    fig.tight_layout()
    save(fig, FIGURES_DIR / "05_accuracy_by_length_and_position.png")
    print("Saved reports/figures/05_accuracy_by_length_and_position.png")


def fig06_latent_space_umap() -> None:
    """Figure 6: UMAP projection of BiLSTM residue representations coloured by DSSP secondary structure state."""
    emb_path = RESULTS_DIR / "cb513_sample_embeddings.pt"
    if not emb_path.exists():
        print("Skipping Fig 6: cb513_sample_embeddings.pt not found.")
        return

    samples = torch.load(emb_path, weights_only=False)
    all_reps = np.concatenate([s["rep"] for s in samples], axis=0)
    all_q8 = np.concatenate([s["q8"] for s in samples], axis=0)
    all_q3 = np.concatenate([s["q3"] for s in samples], axis=0)

    valid = all_q8 != IGNORE_INDEX
    all_reps = all_reps[valid]
    all_q8 = all_q8[valid]
    all_q3 = all_q3[valid]

    # Subsample up to 4,000 residues for clean UMAP
    N = min(len(all_reps), 4000)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(all_reps), size=N, replace=False)
    reps_sub = all_reps[idx]
    q8_sub = all_q8[idx]
    q3_sub = all_q3[idx]

    reducer = umap.UMAP(n_neighbors=25, min_dist=0.3, metric="cosine", random_state=42)
    coords = reducer.fit_transform(reps_sub)

    from sklearn.model_selection import cross_val_score
    sil_score = silhouette_score(coords, q8_sub, sample_size=min(2000, N))
    knn_acc = float(cross_val_score(KNeighborsClassifier(n_neighbors=5), coords, q8_sub, cv=5).mean())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: DSSP 8-State Coloring
    ax = axes[0]
    for q8_idx, q8_char in enumerate(Q8_CHARS):
        mask = (q8_sub == q8_idx)
        if mask.sum() > 0:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=Q8_COLORS[q8_char],
                label=f"{q8_char} (N={mask.sum():,})",
                alpha=0.65,
                s=14,
                edgecolors="none",
            )
    ax.set_title("Residue Latent Space (DSSP 8-State)", pad=10)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(title="DSSP State", frameon=True, loc="upper right", markerscale=2.5)
    panel_label(ax, "A")

    # Panel B: Reduced 3-State (Helix, Sheet, Coil)
    ax = axes[1]
    q3_names = ["H (Helix)", "E (Beta-sheet)", "C (Coil/Loop)"]
    for q3_idx, q3_char in enumerate(Q3_CHARS):
        mask = (q3_sub == q3_idx)
        if mask.sum() > 0:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=Q3_COLORS[q3_char],
                label=f"{q3_names[q3_idx]} (N={mask.sum():,})",
                alpha=0.65,
                s=14,
                edgecolors="none",
            )
    ax.set_title("Residue Latent Space (3-State Reduced)", pad=10)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(title="Class", frameon=True, loc="upper right", markerscale=2.5)
    panel_label(ax, "B")

    fig.suptitle(
        f"CNN-BiLSTM Learned Hidden Residue Embeddings (CB513 Held-Out Benchmark)\n"
        f"N={N:,} residues | Cosine UMAP | 5-CV 5-NN Q8 Acc = {knn_acc*100:.1f}% | Silhouette Score = {sil_score:.3f}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    save(fig, FIGURES_DIR / "06_latent_space_umap.png")
    print("Saved reports/figures/06_latent_space_umap.png")


def main() -> None:
    print("Generating secondary structure figures...")
    fig01_dataset_overview()
    fig02_model_benchmark_comparison()
    fig03_confusion_matrix_q8_cb513()
    fig04_per_residue_prediction_track()
    fig05_accuracy_by_length_and_position()
    fig06_latent_space_umap()
    print("Figure generation complete!")


if __name__ == "__main__":
    main()
