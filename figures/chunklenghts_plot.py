#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ridgeplot de tamaÃ±os de chunks (ChromoPainter *.chunklengths.out)

- Input:
    --haploid  *.chunklengths.out
    --diploid  *.chunklengths.out
    --colors   colors_pop.tsv
- ZTCN colapsado (ZTCN*)
- Orden fijo Nâ†’S con norte arriba
- Colores tomados desde colors_pop.tsv
- Misma escala horizontal haploide/diploide
- Salida: PDF con panel combinado

Uso:
python ridge_chunklengths_NS_colors.py \
  --haploid haploid.chunklengths.out \
  --diploid diploid.chunklengths.out \
  --colors  colors_pop.tsv \
  --out     ridge_chunklengths_NS.pdf
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# -----------------------------
# Orden fijo N â†’ S (ZTCN colapsado)
# -----------------------------
POP_ORDER = ["POS", "QCZ", "ZTCN", "RIT", "CONS", "CVG", "LIL", "LLI", "LOB", "PUC"]


# -----------------------------
# Parsing de poblaciÃ³n
# -----------------------------
def extract_pop(recipient_id: str) -> str:
    if recipient_id.startswith("ZTCN"):
        return "ZTCN"
    m = re.match(r"([A-Za-z]+)", recipient_id)
    return m.group(1) if m else "UNK"


# -----------------------------
# Leer colores desde TSV
# -----------------------------
def read_colors(path: str) -> dict:
    df = pd.read_csv(path, sep=r"\s+", header=0)
    if df.shape[1] < 2:
        raise ValueError("colors_pop.tsv debe tener al menos 2 columnas: POP y COLOR")
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


# -----------------------------
# Leer chunklengths.out
# -----------------------------
def read_chunklengths(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        comment="#",
        header=0
    )
    if "Recipient" not in df.columns:
        raise ValueError(f"No se encuentra columna 'Recipient' en {path}")

    long = df.melt(id_vars=["Recipient"], var_name="Donor", value_name="Length")
    long["Length"] = pd.to_numeric(long["Length"], errors="coerce")
    long = long.dropna()
    long = long[long["Length"] > 0]
    long["pop"] = long["Recipient"].apply(extract_pop)
    return long


# -----------------------------
# Ridgeplot
# -----------------------------
def ridge_panel(ax, long, pops, colors, x_min, x_max, title):
    xg = np.linspace(np.log10(x_min), np.log10(x_max), 500)
    n = len(pops)
    height = 0.85

    for i, p in enumerate(pops):
        y0 = (n - 1 - i)  # norte arriba
        d = long.loc[long["pop"] == p, "Length"].values

        if d.size < 5:
            ax.hlines(y0, x_min, x_max, color=colors.get(p, "0.6"), alpha=0.4)
            ax.text(x_min * 0.85, y0, p, va="center", ha="right", fontsize=9)
            continue

        logd = np.log10(d)
        kde = gaussian_kde(logd)
        y = kde(xg)
        y = (y / y.max()) * height

        ax.fill_between(
            10 ** xg,
            y0 + y,
            y0,
            color=colors.get(p, "0.6"),
            alpha=0.9,
            linewidth=0
        )
        ax.text(x_min * 0.85, y0, p, va="center", ha="right", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel("Chunk length (log scale)")
    ax.set_title(title)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--haploid", required=True)
    ap.add_argument("--diploid", required=True)
    ap.add_argument("--colors", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    colors = read_colors(args.colors)
    hap = read_chunklengths(args.haploid)
    dip = read_chunklengths(args.diploid)

    pops = [p for p in POP_ORDER if p in set(hap["pop"]) or p in set(dip["pop"])]

    all_lengths = np.concatenate([hap["Length"].values, dip["Length"].values])
    x_min = max(np.quantile(all_lengths, 0.001), 1e-6)
    x_max = np.quantile(all_lengths, 0.999)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ridge_panel(
        axes[0], hap, pops, colors, x_min, x_max,
        "Haploid â€“ chunk lengths (ZTCN colapsado, Nâ†’S)"
    )
    ridge_panel(
        axes[1], dip, pops, colors, x_min, x_max,
        "Diploid â€“ chunk lengths (ZTCN colapsado, Nâ†’S)"
    )

    for ax in axes:
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
