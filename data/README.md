# NetSurfP-2.0 Benchmark Datasets

This directory contains the training and independent test benchmark datasets from **NetSurfP-2.0** ([Klausen et al., *Proteins*, 2019](https://doi.org/10.1002/prot.25674)), derived from the CullPDB non-redundant dataset (<25% sequence identity, resolution < 2.5 Å, R-factor < 1.0).

---

## Benchmark Splits

| Split | Raw File | Sequences | Residues (Raw Array / Valid Mask) | Purpose |
|---|---|---|---|---|
| **Train** | `Train_HHblits.npz` | 10,848 | 2,778,438 / — | Model training & internal validation (`10,348` train / `500` val) |
| **CB513** | `CB513_HHblits.npz` | 513 | 144,360 / 83,772 | De facto gold-standard blind benchmark in secondary structure prediction |
| **TS115** | `TS115_HHblits.npz` | 115 | 29,704 / 27,501 | High-resolution test set strictly non-homologous to training set |
| **CASP12** | `CASP12_HHblits.npz` | 21 | 7,256 / 6,420 | Free-modeling and hard targets from CASP12 competition |

---

## Data Layout & Feature Encoding

Each raw `.npz` archive contains a 3D NumPy array of shape `(N, 2170, 65)` padded to the maximum sequence length (2,170 residues). During preprocessing, sequences are extracted into compact contiguous ragged arrays (`data/processed/*.npz`):

- **Channels `0:20`**: One-hot amino acid identity across 20 standard residues (`A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V`).
- **Channels `20:50`**: Evolutionary profile extracted from HHblits against UniClust30 (30 channels: 20 match probabilities, 7 transition probabilities, 3 diversity metrics).
- **Channel `50`**: Sequence mask (1 if valid residue, 0 if padding).
- **Channel `51`**: Disorder mask (1 if resolved in crystal structure, 0 if disordered).
- **Channel `52`**: Evaluation mask (1 if eligible for DSSP benchmark evaluation, 0 otherwise).
- **Channels `57:65`**: 8-state DSSP ground truth one-hot vector:
  - `G`: $3_{10}$-helix
  - `H`: $\alpha$-helix
  - `I`: $\pi$-helix
  - `B`: Isolated $\beta$-bridge
  - `E`: Extended $\beta$-strand
  - `S`: Bend
  - `T`: Hydrogen-bonded turn
  - `C`: Random coil / unstructured loop

### 3-State Reduction (Q3)

For 3-state evaluation, the standard CASP/Rost-Sander reduction mapping is applied:
- **Helix ($H$)**: `H`, `G`, `I`
- **Strand ($E$)**: `E`, `B`
- **Coil ($C$)**: `S`, `T`, `C`

---

## Training & Preprocessing Entry Point

To process raw NetSurfP-2.0 `.npz` archives into compact ragged splits (`data/processed/*.npz`) and run training:

```bash
python scripts/train.py
```

