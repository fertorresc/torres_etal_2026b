#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ChromoPainter coancestry heatmap (chunkcounts.out) â€” reproducible script.

What it does
- Reads a ChromoPainter/FineSTRUCTURE-style *.chunkcounts.out
- Reads an *.ids file (one sample ID per line; whitespace OK)
- Reorders individuals by an explicit population order (e.g., POS, QCZ, ...)
  and then by sample ID within population
- Produces TWO heatmaps:
    (1) raw chunks
    (2) log1p(chunks)
- Adds population color bars (top + left), separators between populations,
  and an explicit legend.

Usage
  python cp_coancestry_heatmap.py \
    --chunkcounts 600_Diploids_linked.chunkcounts.out \
    --ids 600_Diploids.ids \
    --pop-order POS QCZ ZTCN RIT CONS CVG LLI LIL LOB PUC \
    --out-prefix coancestry_poporder_withlabels \
    --labels

Optional
  --no-labels   (default if --labels not provided)
  --figsize 14
  --dpi 300
"""

import argparse
import os
import re
import sys
import subprocess

# -------------------------
# Dependency bootstrap
# -------------------------
REQUIRED = ["numpy", "pandas", "matplotlib"]

def _pip_install(packages):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    print("[INFO] Installing missing packages:", " ".join(packages), file=sys.stderr)
    subprocess.check_call(cmd)

def ensure_deps():
    missing = []
    for pkg in REQUIRED:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        _pip_install(missing)

ensure_deps()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------------
# Core helpers
# -------------------------
def read_ids(ids_path):
    df = pd.read_csv(ids_path, header=None, sep=r"\s+", dtype=str)
    if df.shape[1] < 1:
        raise ValueError("IDs file appears empty or malformed.")
    return df.iloc[:, 0].astype(str).tolist()

def read_chunkcounts(chunkcounts_path, ids):
    """
    ChromoPainter chunkcounts.out structure (typical):
      line 1: #Cfactor ...
      line 2: Recipient <id1> <id2> ...
      line 3+: <donor_id> <v1> <v2> ...

    We skip first 2 lines and parse whitespace.
    """
    df = pd.read_csv(chunkcounts_path, sep=r"\s+", skiprows=2, header=None, dtype=str)
    donors = df.iloc[:, 0].astype(str).tolist()
    mat = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()

    if mat.shape[1] != len(ids):
        raise ValueError(
            f"Column mismatch: matrix has {mat.shape[1]} recipients, ids has {len(ids)}. "
            "Check that *.ids matches the Recipient header in chunkcounts.out."
        )

    out = pd.DataFrame(mat, index=donors, columns=ids)
    # Basic sanity check: donors should match ids for square matrices
    if out.shape[0] != out.shape[1]:
        print(f"[WARN] Non-square matrix: {out.shape[0]}x{out.shape[1]}. Proceeding anyway.", file=sys.stderr)
    return out

def pop_code(sample_id):
    """
    Extract population code from sample ID.
    Default: leading letters; special-case ZTCN* -> ZTCN.
    Customize if your naming scheme differs.
    """
    s = str(sample_id)
    if s.startswith("ZTCN"):
        return "ZTCN"
    m = re.match(r"([A-Za-z]+)", s)
    return m.group(1) if m else s

def build_order(samples, pop_order):
    """
    Order = pop_order groups, then by sample ID.
    Unknown pops go to the end.
    """
    rank = {p: i for i, p in enumerate(pop_order)}
    df = pd.DataFrame({"sample": samples})
    df["pop"] = df["sample"].map(pop_code)
    df["pop_rank"] = df["pop"].map(rank).fillna(len(pop_order)).astype(int)
    df = df.sort_values(["pop_rank", "sample"])
    return df["sample"].tolist()

def make_pop_palette(pop_order):
    base = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {p: base[i % len(base)] for i, p in enumerate(pop_order)}

def plot_heatmap(mat_df, order, pop_order, out_prefix,
                 with_log=True, labels=False, figsize=14, dpi=300):
    # Reorder
    mat_ord = mat_df.loc[order, order]

    vals = mat_ord.to_numpy(dtype=float)
    if with_log:
        vals = np.log1p(vals)
        cbar_label = "log1p(chunks)"
        suffix = "log1p"
    else:
        cbar_label = "chunks"
        suffix = "raw"

    pops = [pop_code(s) for s in order]
    pop_colors = make_pop_palette(pop_order)

    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111)

    im = ax.imshow(vals, aspect="equal", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    n = len(order)
    if labels:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(order, rotation=90, fontsize=5)
        ax.set_yticklabels(order, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel("Recipients")
    ax.set_ylabel("Donors")
    ax.set_title(
        f"Coancestry matrix â€” ordered: {', '.join(pop_order)} ({suffix})"
    )

    # Pop bars (top + left)
    bar = max(1, int(n * 0.01))
    for j, p in enumerate(pops):
        col = pop_colors.get(p, "grey")
        ax.add_patch(Rectangle((j - 0.5, -0.5 - bar), 1, bar, color=col, clip_on=False, linewidth=0))
    for i, p in enumerate(pops):
        col = pop_colors.get(p, "grey")
        ax.add_patch(Rectangle((-0.5 - bar, i - 0.5), bar, 1, color=col, clip_on=False, linewidth=0))

    # Separators where population changes
    for i in range(1, n):
        if pops[i] != pops[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=0.8)
            ax.axvline(i - 0.5, color="white", linewidth=0.8)

    # Legend
    handles = [Rectangle((0, 0), 1, 1, color=pop_colors[p]) for p in pop_order]
    ax.legend(
        handles, pop_order, title="Pop order", loc="upper right",
        bbox_to_anchor=(1.34, 1.02), frameon=False, fontsize=8, title_fontsize=9
    )

    plt.tight_layout()

    pdf = f"{out_prefix}_{suffix}.pdf"
    svg = f"{out_prefix}_{suffix}.svg"
    png = f"{out_prefix}_{suffix}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf, svg, png

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot ChromoPainter coancestry heatmaps (raw and log1p)."
    )
    ap.add_argument("--chunkcounts", required=True, help="Path to *.chunkcounts.out")
    ap.add_argument("--ids", required=True, help="Path to *.ids (one sample per line)")
    ap.add_argument("--pop-order", nargs="+", required=True,
                    help="Population order list, e.g. POS QCZ ZTCN RIT CONS CVG LLI LIL LOB PUC")
    ap.add_argument("--out-prefix", default="coancestry_heatmap",
                    help="Output prefix (without extension)")
    ap.add_argument("--labels", action="store_true", help="Show individual labels (recommended for n<=120)")
    ap.add_argument("--figsize", type=float, default=14.0, help="Figure size in inches (square)")
    ap.add_argument("--dpi", type=int, default=300, help="PNG dpi")
    return ap.parse_args()

def main():
    args = parse_args()

    ids = read_ids(args.ids)
    mat = read_chunkcounts(args.chunkcounts, ids)

    order = build_order(mat.index.tolist(), args.pop_order)

    # Enforce same order exists in columns (should, for square matrices)
    missing_rows = [x for x in order if x not in mat.index]
    missing_cols = [x for x in order if x not in mat.columns]
    if missing_rows or missing_cols:
        print("[WARN] Missing samples detected after ordering.", file=sys.stderr)
        if missing_rows:
            print("  Missing in rows:", missing_rows[:10], file=sys.stderr)
        if missing_cols:
            print("  Missing in cols:", missing_cols[:10], file=sys.stderr)

    # Plot both versions
    raw = plot_heatmap(
        mat, order, args.pop_order,
        out_prefix=args.out_prefix,
        with_log=False,
        labels=args.labels,
        figsize=args.figsize,
        dpi=args.dpi
    )
    log = plot_heatmap(
        mat, order, args.pop_order,
        out_prefix=args.out_prefix,
        with_log=True,
        labels=args.labels,
        figsize=args.figsize,
        dpi=args.dpi
    )

    print("[OK] Wrote files:")
    for p in raw + log:
        print(" ", os.path.abspath(p))

if __name__ == "__main__":
    main()
