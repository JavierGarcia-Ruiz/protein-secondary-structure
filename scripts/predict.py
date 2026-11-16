"""Single-sequence CLI for protein secondary structure prediction.

Usage:
    python scripts/predict.py --sequence "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    python scripts/predict.py --fasta path/to/protein.fasta --model models/cnn_bilstm_profile.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from ssp.data.constants import AA_TO_IDX, Q8_CHARS, Q3_CHARS
from ssp.models.factory import build_model
from ssp.utils import load_config


def predict_sequence(
    sequence: str,
    model_path: str = "models/cnn_bilstm_profile.pt",
    config_path: str = "configs/default.yaml",
    device: str | None = None,
) -> dict[str, str | list[float]]:
    """Predict Q8 and Q3 secondary structure for an amino acid sequence."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Validate sequence
    seq = sequence.strip().upper()
    L = len(seq)
    if L == 0:
        raise ValueError("Input sequence cannot be empty.")

    # Load configuration
    cfg = load_config(config_path)
    
    # Detect feature dimension from checkpoint or config
    ckpt = torch.load(model_path, map_location=dev, weights_only=True)
    in_weight = ckpt.get("input_proj.weight")
    if in_weight is not None:
        d_in = in_weight.shape[1]
    else:
        d_in = 50

    cfg["model"]["name"] = "cnn_bilstm"
    cfg["model"]["d_in"] = d_in

    # Build and load model
    model = build_model(cfg)
    model.load_state_dict(ckpt)
    model.to(dev)
    model.eval()

    # Build input tensor
    # 0..20: One-hot (20 amino acids)
    # 20..50: Profile (if 50-D, fill profile with BLOSUM/uniform background when raw profile absent)
    feats = np.zeros((L, d_in), dtype=np.float32)
    for i, aa in enumerate(seq):
        idx = AA_TO_IDX.get(aa, -1)
        if 0 <= idx < 20:
            feats[i, idx] = 1.0

    if d_in == 50:
        # Background amino acid distribution approximation for zero-profile inference
        bg = np.array([
            0.078, 0.052, 0.044, 0.054, 0.015, 0.038, 0.068, 0.071, 0.022, 0.057,
            0.095, 0.058, 0.024, 0.040, 0.047, 0.065, 0.055, 0.012, 0.032, 0.067
        ], dtype=np.float32)
        feats[:, 20:40] = bg
        feats[:, 40:50] = 0.0

    x = torch.from_numpy(feats).unsqueeze(0).to(dev)
    mask = torch.ones(1, L, dtype=torch.bool, device=dev)
    lengths = torch.tensor([L], device=dev)

    with torch.no_grad():
        out = model(x, mask=mask, lengths=lengths)
        probs_q8 = torch.softmax(out["q8_logits"], dim=-1)[0].cpu().numpy()
        pred_q8_idx = np.argmax(probs_q8, axis=-1)
        probs_q3 = torch.softmax(out["q3_logits"], dim=-1)[0].cpu().numpy()
        pred_q3_idx = np.argmax(probs_q3, axis=-1)

    pred_q8_str = "".join([Q8_CHARS[i] for i in pred_q8_idx])
    pred_q3_str = "".join([Q3_CHARS[i] for i in pred_q3_idx])

    return {
        "sequence": seq,
        "length": L,
        "q8": pred_q8_str,
        "q3": pred_q3_str,
        "probs_q8": probs_q8.tolist(),
        "probs_q3": probs_q3.tolist(),
    }


def format_prediction_ascii(seq: str, q8: str, q3: str, line_width: int = 60) -> str:
    """Format predictions in aligned sequence blocks."""
    lines = []
    L = len(seq)
    for start in range(0, L, line_width):
        end = min(start + line_width, L)
        pos_header = f"Residues {start + 1:4d} - {end:4d}"
        lines.append(f"{pos_header}:")
        lines.append(f"  Seq: {seq[start:end]}")
        lines.append(f"  Q8 : {q8[start:end]}")
        lines.append(f"  Q3 : {q3[start:end]}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict protein secondary structure")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", type=str, help="Amino acid sequence string")
    group.add_argument("--fasta", type=str, help="Path to input FASTA file")
    parser.add_argument("--model", type=str, default="models/cnn_bilstm_profile.pt", help="Path to trained checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    if args.fasta:
        with open(args.fasta, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if not l.startswith(">")]
            seq = "".join(lines)
    else:
        seq = args.sequence

    res = predict_sequence(seq, model_path=args.model, config_path=args.config, device=args.device)

    print("\n" + "=" * 60)
    print(f"PROTEIN SECONDARY STRUCTURE PREDICTION (L={res['length']})")
    print("=" * 60)
    print(format_prediction_ascii(res["sequence"], res["q8"], res["q3"]))
    print("DSSP Q8 Notation:")
    print("  H: Alpha-helix   | G: 3_10-helix    | I: Pi-helix")
    print("  E: Beta-strand   | B: Beta-bridge   | T: Turn")
    print("  S: Bend          | C: Coil/Loop")
    print("=" * 60)


if __name__ == "__main__":
    main()

