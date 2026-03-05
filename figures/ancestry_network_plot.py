#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ChromoPainter/FineSTRUCTURE coancestry networks (chunkcounts) Ã¢â‚¬â€œ multi-threshold SVG/PDF exporter.

Input
  - *.chunkcounts.out (ChromoPainter/FineSTRUCTURE-style)
  - Optional: population mapping via prefixes (default) or a popmap file

What it does
  1) Reads the chunkcounts matrix (Recipients x Donors; typically same IDs)
  2) Aggregates to POPULATION x POPULATION by summing chunks
  3) Normalizes per-recipient population (row-wise) to proportions
  4) Builds an undirected network where edge weight w = mean( p(i->j), p(j->i) )
  5) For each threshold, keeps edges with w > threshold
  6) Draws network with spring layout where distances reflect 1/w (stronger = closer)
  7) Saves one SVG per dataset per threshold, plus combined multipanel SVGs

Usage example
  python cp_network_thresholds.py \
    --haploid example_linked.chunkcounts.out \
    --diploid diploid_noZ_linked.chunkcounts.out \
    --thresholds 0.02 0.03 0.04 0.045 0.05 \
    --outdir networks_out \
    --seed 7 \
    --k 1.2

Optional: custom colors (population -> hex)
  python cp_network_thresholds.py \
    --haploid ... --diploid ... \
    --colors colors_pop.tsv

colors_pop.tsv format (tab-delimited):
  POP   HEX
  POS   #FF0000
  QCZ   #E8A8A8
  ...

Optional: popmap for exact pop assignment (recommended if prefixes are messy)
  --popmap popmap.tsv

popmap.tsv format (tab-delimited):
  ID    POP
  sample1 POS
  sample2 POS
  ...

Notes
  - Default population extraction mimics your case:
      ZTCN_para*, ZTCN03N*, ZTCN03S* are preserved,
      else first letters prefix (e.g., CONS01 -> CONS).
  - Output is vector (SVG). You can also export PDF with --pdf.
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# -----------------------------
# Population parsing
# -----------------------------
def extract_pop_from_id(sample_id: str) -> str:
    """Default rule-based population parser."""
    if sample_id.startswith("ZTCN_para"):
        return "ZTCN_para"
    if sample_id.startswith("ZTCN03N"):
        return "ZTCN03N"
    if sample_id.startswith("ZTCN03S"):
        return "ZTCN03S"
    m = re.match(r"([A-Za-z]+)", sample_id)
    return m.group(1) if m else sample_id


def load_popmap(path: str) -> Dict[str, str]:
    """Load ID->POP mapping from a 2-column TSV."""
    df = pd.read_csv(path, sep="\t", header=None, names=["id", "pop"])
    if df.isna().any().any():
        raise ValueError(f"[popmap] Missing values detected in: {path}")
    return dict(zip(df["id"].astype(str), df["pop"].astype(str)))


# -----------------------------
# Chunkcounts loading
# -----------------------------
def load_chunkcounts(path: str) -> pd.DataFrame:
    """
    Read ChromoPainter/FineSTRUCTURE-style *.chunkcounts.out as a square matrix.
    Assumes:
      - Line 2 has the header with 'Recipient' + donor IDs
      - Data begins line 3, first col is recipient ID
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    if len(lines) < 3:
        raise ValueError(f"[chunkcounts] File too short: {path}")

    header_tokens = lines[1].strip().split()
    if "Recipient" not in header_tokens[0]:
        # Some outputs include 'Recipient' as first token; if not, still proceed
        pass

    # Number of recipient rows is typically len(header_tokens)-1 (excluding 'Recipient')
    nrows = len(header_tokens) - 1
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=2,
        header=None,
        nrows=nrows,
    )
    df.columns = header_tokens
    df = df.set_index("Recipient")
    df = df.apply(pd.to_numeric, errors="coerce")

    # Keep only donors that also appear as recipients (square)
    donors_in_recipients = [c for c in df.columns if c in df.index]
    df = df.loc[df.index, donors_in_recipients]

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"[chunkcounts] Empty matrix after parsing: {path}")

    return df


# -----------------------------
# Population aggregation + normalization
# -----------------------------
def aggregate_to_populations(
    mat: pd.DataFrame,
    popmap: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Aggregate individual matrix to POPxPOP by summing chunks.
    """
    ids = mat.index.astype(str)

    if popmap is None:
        id_to_pop = {i: extract_pop_from_id(i) for i in ids}
    else:
        missing = [i for i in ids if i not in popmap]
        if missing:
            raise ValueError(
                f"[popmap] Missing {len(missing)} IDs from popmap. Example: {missing[:5]}"
            )
        id_to_pop = {i: popmap[i] for i in ids}

    # rows: recipients -> pop
    df = mat.copy()
    df["pop"] = [id_to_pop[i] for i in df.index]
    pop_rows = df.groupby("pop").sum()

    # cols: donors -> pop
    pop_cols = pop_rows.T.copy()
    pop_cols["pop"] = [id_to_pop[i] for i in pop_cols.index]
    pop_mat = pop_cols.groupby("pop").sum().T

    # Ensure numeric
    pop_mat = pop_mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return pop_mat


def row_normalize(pop_mat: pd.DataFrame) -> pd.DataFrame:
    """Row-wise normalize to proportions (recipient-based)."""
    row_sums = pop_mat.sum(axis=1).replace(0, np.nan)
    norm = pop_mat.div(row_sums, axis=0).fillna(0.0)
    return norm


def symmetric_weight(norm: pd.DataFrame, a: str, b: str) -> float:
    """w = mean( p(a->b), p(b->a) )"""
    return float((norm.loc[a, b] + norm.loc[b, a]) / 2.0)


# -----------------------------
# Network building + plotting
# -----------------------------
def build_graph(norm: pd.DataFrame, threshold: float) -> nx.Graph:
    """
    Build undirected graph with edges for w>threshold.
    Layout uses inverse weights so stronger links are shorter.
    """
    pops = norm.index.tolist()
    G = nx.Graph()
    for p in pops:
        G.add_node(p)

    for i, a in enumerate(pops):
        for j in range(i + 1, len(pops)):
            b = pops[j]
            w = symmetric_weight(norm, a, b)
            if w > threshold:
                # for layout distance: use 1/w
                G.add_edge(a, b, w=w, inv_w=1.0 / (w + 1e-9))
    return G


def load_colors(path: str) -> Dict[str, str]:
    """
    Load POP->HEX from a 2-column TSV (POP <tab> HEX).
    """
    df = pd.read_csv(path, sep="\t", header=None, names=["pop", "hex"])
    return dict(zip(df["pop"].astype(str), df["hex"].astype(str)))


def draw_graph(
    ax,
    G: nx.Graph,
    colors: Dict[str, str],
    seed: int,
    k: float,
    title: str,
    show_edge_labels: bool = True,
    node_size: int = 2000,
    edge_scale: float = 10.0,
    decimals: int = 3,
):
    # Layout: use inv_w so larger w => smaller distance
    pos = nx.spring_layout(G, seed=seed, k=k, weight="inv_w")

    nodes = list(G.nodes())
    node_colors = [colors.get(n, "#88ccee") for n in nodes]  # fallback

    edges = list(G.edges(data=True))
    widths = [e[2]["w"] * edge_scale for e in edges]

    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="black")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)

    if show_edge_labels and len(edges) > 0:
        edge_labels = {(u, v): f"{d['w']:.{decimals}f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=10)

    ax.set_title(title)
    ax.axis("off")


# -----------------------------
# Main
# -----------------------------
def make_multipanel(
    outpath: str,
    label: str,
    norms: Dict[str, pd.DataFrame],
    thresholds: List[float],
    colors: Dict[str, str],
    seed: int,
    k: float,
    pdf: bool,
    show_edge_labels: bool,
):
    """
    Create a multipanel figure with rows = thresholds, cols = datasets (haploid/diploid/...).
    """
    datasets = list(norms.keys())
    nrows = len(thresholds)
    ncols = len(datasets)

    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 5.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    for r, thr in enumerate(thresholds):
        for c, ds in enumerate(datasets):
            ax = axes[r, c]
            norm = norms[ds]
            G = build_graph(norm, thr)
            draw_graph(
                ax,
                G,
                colors=colors,
                seed=seed,
                k=k,
                title=f"{ds} | thr={thr}",
                show_edge_labels=show_edge_labels,
            )

    fig.suptitle(label, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if outpath.lower().endswith(".svg"):
        plt.savefig(outpath, format="svg")
    elif outpath.lower().endswith(".pdf"):
        plt.savefig(outpath, format="pdf")
    else:
        plt.savefig(outpath)

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--haploid", help="Haploid *.chunkcounts.out", default=None)
    ap.add_argument("--diploid", help="Diploid *.chunkcounts.out", default=None)
    ap.add_argument("--inputs", nargs="*", default=None,
                    help="Optional: any number of label=path entries, e.g. HAP=foo.out DIP=bar.out")
    ap.add_argument("--popmap", default=None, help="Optional ID->POP TSV (2 cols: ID, POP)")
    ap.add_argument("--colors", default=None, help="Optional POP->HEX TSV (2 cols: POP, HEX)")
    ap.add_argument("--thresholds", nargs="+", type=float, required=True,
                    help="Threshold list, e.g. 0.02 0.03 0.04 0.045 0.05")
    ap.add_argument("--outdir", default="networks_out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=float, default=1.2)
    ap.add_argument("--pdf", action="store_true", help="Also export PDF multipanel")
    ap.add_argument("--no-edge-labels", action="store_true", help="Disable edge labels")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Collect datasets
    datasets: List[Tuple[str, str]] = []
    if args.haploid:
        datasets.append(("haploid", args.haploid))
    if args.diploid:
        datasets.append(("diploid", args.diploid))
    if args.inputs:
        for item in args.inputs:
            if "=" not in item:
                raise ValueError(f"--inputs entries must be label=path. Got: {item}")
            lab, path = item.split("=", 1)
            datasets.append((lab, path))

    if not datasets:
        raise ValueError("Provide at least one input via --haploid/--diploid or --inputs label=path ...")

    popmap = load_popmap(args.popmap) if args.popmap else None

    # Colors
    colors = {}
    if args.colors:
        colors = load_colors(args.colors)

    # Compute normalized POPxPOP for each dataset
    norms: Dict[str, pd.DataFrame] = {}
    for lab, path in datasets:
        mat = load_chunkcounts(path)
        pop_mat = aggregate_to_populations(mat, popmap=popmap)
        norms[lab] = row_normalize(pop_mat)

    # Export per-threshold per-dataset SVG
    for thr in args.thresholds:
        for lab in norms.keys():
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            G = build_graph(norms[lab], thr)
            draw_graph(
                ax,
                G,
                colors=colors,
                seed=args.seed,
                k=args.k,
                title=f"{lab} | threshold={thr}",
                show_edge_labels=(not args.no_edge_labels),
            )
            out_svg = os.path.join(args.outdir, f"network_{lab}_thr{str(thr).replace('.','p')}.svg")
            plt.tight_layout()
            plt.savefig(out_svg, format="svg")
            plt.close(fig)

    # Export multipanel SVG (all thresholds x all datasets)
    out_multi_svg = os.path.join(args.outdir, "networks_multipanel.svg")
    make_multipanel(
        outpath=out_multi_svg,
        label="Coancestry networks (chunkcounts) Ã¢â‚¬â€œ multiple thresholds",
        norms=norms,
        thresholds=args.thresholds,
        colors=colors,
        seed=args.seed,
        k=args.k,
        pdf=False,
        show_edge_labels=(not args.no_edge_labels),
    )

    if args.pdf:
        out_multi_pdf = os.path.join(args.outdir, "networks_multipanel.pdf")
        make_multipanel(
            outpath=out_multi_pdf,
            label="Coancestry networks (chunkcounts) Ã¢â‚¬â€œ multiple thresholds",
            norms=norms,
            thresholds=args.thresholds,
            colors=colors,
            seed=args.seed,
            k=args.k,
            pdf=True,
            show_edge_labels=(not args.no_edge_labels),
        )

    print(f"[OK] Wrote outputs to: {args.outdir}")
    print(" - Per-threshold SVGs: network_<label>_thr*.svg")
    print(" - Multipanel SVG: networks_multipanel.svg")
    if args.pdf:
        print(" - Multipanel PDF: networks_multipanel.pdf")


if __name__ == "__main__":
    main()
