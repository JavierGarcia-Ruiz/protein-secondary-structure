"""Generate clean Jupyter notebooks for Protein Secondary Structure."""

import json
from pathlib import Path


def create_eda_notebook(path: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 01. Exploratory Data Analysis — NetSurfP-2.0 Benchmark Suite\n",
                    "\n",
                    "This notebook explores the sequence characteristics, amino acid distributions, and secondary structure class balances across the **NetSurfP-2.0** training corpus and independent test sets (**CB513**, **TS115**, **CASP12**)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from ssp.data.netsurfp import parse_netsurfp\n",
                    "from ssp.data.constants import Q8_CHARS, Q3_CHARS, AA_CHARS\n",
                    "from ssp.visualization.plotting import set_style, Q8_COLORS, OKABE_ITO\n",
                    "\n",
                    "set_style()\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Processed Ragged Splits"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "splits = {\n",
                    "    'train': parse_netsurfp('data/processed/train.npz'),\n",
                    "    'cb513': parse_netsurfp('data/processed/cb513.npz'),\n",
                    "    'ts115': parse_netsurfp('data/processed/ts115.npz'),\n",
                    "    'casp12': parse_netsurfp('data/processed/casp12.npz')\n",
                    "}\n",
                    "\n",
                    "summary = []\n",
                    "for name, s in splits.items():\n",
                    "    lengths = s.lengths\n",
                    "    summary.append({\n",
                    "        'Split': name.upper(),\n",
                    "        'Sequences': len(s),\n",
                    "        'Residues': len(s.features),\n",
                    "        'Mean Length': np.mean(lengths),\n",
                    "        'Median Length': np.median(lengths),\n",
                    "        'Min Length': np.min(lengths),\n",
                    "        'Max Length': np.max(lengths),\n",
                    "    })\n",
                    "df_summary = pd.DataFrame(summary)\n",
                    "df_summary"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. DSSP Q8 Class Frequencies"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, ax = plt.subplots(figsize=(10, 4))\n",
                    "q8_tr = np.bincount(splits['train'].labels_q8, minlength=8) / len(splits['train'].labels_q8) * 100\n",
                    "q8_cb = np.bincount(splits['cb513'].labels_q8, minlength=8) / len(splits['cb513'].labels_q8) * 100\n",
                    "\n",
                    "x = np.arange(8)\n",
                    "w = 0.35\n",
                    "ax.bar(x - w/2, q8_tr, width=w, label='Train', color='#0072B2')\n",
                    "ax.bar(x + w/2, q8_cb, width=w, label='CB513', color='#D55E00')\n",
                    "ax.set_xticks(x)\n",
                    "ax.set_xticklabels(Q8_CHARS)\n",
                    "ax.set_ylabel('Frequency (%)')\n",
                    "ax.set_title('Q8 Secondary Structure Distribution (Train vs CB513)')\n",
                    "ax.legend()\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def create_results_notebook(path: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 02. Benchmark Results & Error Analysis\n",
                    "\n",
                    "This notebook evaluates model architectures (CNN, CNN-BiLSTM, BiLSTM-CRF) and input features (evolutionary HMM profiles vs one-hot sequences) across three gold-standard benchmarks:\n",
                    "- **CB513** (513 chains)\n",
                    "- **TS115** (115 high-resolution non-homologous chains)\n",
                    "- **CASP12** (21 blind competition targets)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "from pathlib import Path\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from ssp.data.constants import Q8_CHARS, Q3_CHARS\n",
                    "from ssp.visualization.plotting import set_style, OKABE_ITO\n",
                    "\n",
                    "set_style()\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Evaluation Metrics"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_bench = pd.read_csv('reports/results/metrics_benchmarks.csv')\n",
                    "df_bench"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Pivot Table Summary (Q8 and Q3 Accuracy)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pivot_q8 = df_bench.pivot(index='experiment', columns='benchmark', values='q8_acc')\n",
                    "pivot_q3 = df_bench.pivot(index='experiment', columns='benchmark', values='q3_acc')\n",
                    "print('Q8 Accuracy:')\n",
                    "display(pivot_q8)\n",
                    "print('\\nQ3 Accuracy:')\n",
                    "display(pivot_q3)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Confusion Matrix Analysis (CB513)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cm = np.load('reports/results/cm_q8_cb513.npy')\n",
                    "cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100\n",
                    "fig, ax = plt.subplots(figsize=(7, 6))\n",
                    "sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', xticklabels=Q8_CHARS, yticklabels=Q8_CHARS, ax=ax)\n",
                    "ax.set_title('CB513 DSSP Q8 Confusion Matrix (Recall %)')\n",
                    "ax.set_xlabel('Predicted')\n",
                    "ax.set_ylabel('True')\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


if __name__ == "__main__":
    create_eda_notebook(Path("notebooks/01_eda.ipynb"))
    create_results_notebook(Path("notebooks/02_results_analysis.ipynb"))
    print("Created notebooks!")

