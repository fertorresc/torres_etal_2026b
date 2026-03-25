#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import diempy as diem


# =========================================================
# CONFIGURACIÃ“N GENERAL
# =========================================================

BASE_DIR = Path("/media/server/a77f75fe-fd07-402e-84d7-a7341c29141c/fertorres/segundo_capitulo/diemPy/HAP_diempy").resolve()

SOURCE_VCF = BASE_DIR / "vcf_maf_md_600kpb_contigs_HAPLOIDE.vcf.gz"
POPMAP = BASE_DIR / "PopMap_600_HAP.txt"

POPULATIONS = ["QCZ", "ZTCN" , "RIT"]

DATASET_TAG = "HAP_QCZ_ZTCN_RIT"
OUTDIR = BASE_DIR / f"subset_{DATASET_TAG}"
OUTDIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = OUTDIR

SAMPLES_FILE = RUN_DIR / f"samples_{DATASET_TAG}.txt"
VCF = RUN_DIR / f"vcf_{DATASET_TAG}.vcf.gz"

NCORES = 16
RUN_SUBSET = True
RUN_VCF2DIEM = True

# ParÃ¡metros DIEM
DI_THRESHOLD = -50
SMOOTH_SCALES = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
FINAL_SMOOTH_SCALE = 5e-5

# Auto-ajuste desde tablas
AUTO_SELECT_SMOOTH = True
AUTO_SELECT_LEFT_RIGHT = True

SMOOTH_RELATIVE_DELTA_THRESHOLD = 0.05
SMOOTH_MIN_SCALE = 1e-7
MIN_EDGE_BUFFER = 2

# Opcional: si existe esta tabla, el script usa DI_THRESHOLD desde ahÃ­
DI_TOUCHDOWN_TABLE = RUN_DIR / f"{DATASET_TAG}_tables" / f"{DATASET_TAG}.DI_touchdown_scan.tsv"
FALLBACK_DI_THRESHOLD = DI_THRESHOLD
TOUCHDOWN_TOL = 0.02
MIN_RETAINED_FRACTION = 0.02

# Whole-genome plots
GENOME_PIXELS = 360 * 10
TICKS = 100000

# Plots
CONTIGS_TO_PLOT = None
MULTI_CHR_LIST = None
CONTIG_CHUNK_SIZE = 4

# Intermedios por HI
INTERMEDIATE_HI_MIN = 0.20
INTERMEDIATE_HI_MAX = 0.80
USE_MANUAL_INTERMEDIATES = False
MANUAL_INTERMEDIATES = []

# Archivos del pipeline DIEM
META_RAW = RUN_DIR / f"{VCF.name}.diem_meta.bed"
BED_IN = RUN_DIR / f"{VCF.name}.diem_input.bed"
EXCLUDE_BED = RUN_DIR / f"{VCF.name}.diem_exclude.bed"

META_CORRECTED = RUN_DIR / f"{DATASET_TAG}.diem_meta_corrected.bed"
PLOIDY_FILE = RUN_DIR / f"{DATASET_TAG}.ploidy_update.txt"

BED_POL = RUN_DIR / f"{DATASET_TAG}.polarized_output.bed"
DIEMTYPE_POL = RUN_DIR / f"{DATASET_TAG}.polarized.diemtype"
DIEMTYPE_THRESH = RUN_DIR / f"{DATASET_TAG}.thresholded.diemtype"
DIEMTYPE_SMOOTH = RUN_DIR / f"{DATASET_TAG}.smoothed.diemtype"

# Salidas bÃ¡sicas
PLOT_DIR = RUN_DIR / f"{DATASET_TAG}_plots"
CONTIG_DIR = RUN_DIR / f"{DATASET_TAG}_contigs"
TABLES_IN = RUN_DIR / f"{DATASET_TAG}_tables"

# Salidas extendidas
OUTPLOT_DIR = RUN_DIR / f"{DATASET_TAG}_plots_extended"
PAINT_DIR = OUTPLOT_DIR / "paintings"
SUMMARY_DIR = OUTPLOT_DIR / "summary_plots"
TRACT_DIR = OUTPLOT_DIR / "tract_plots"
TABLE_DIR = OUTPLOT_DIR / "tables"
INTERACTIVE_DIR = OUTPLOT_DIR / "interactive_plots"
WHOLEGENOME_DIR = OUTPLOT_DIR / "whole_genome_plots"
CONTIG_CONTRIB_DIR = OUTPLOT_DIR / "contig_contributions"
IND_CONTRIB_DIR = OUTPLOT_DIR / "individual_contributions"

for d in [
    PLOT_DIR, CONTIG_DIR, TABLES_IN,
    OUTPLOT_DIR, PAINT_DIR, SUMMARY_DIR, TRACT_DIR, TABLE_DIR,
    INTERACTIVE_DIR, WHOLEGENOME_DIR, CONTIG_CONTRIB_DIR, IND_CONTRIB_DIR
]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def run_cmd(cmd, cwd=None, capture_output=False):
    print(f"\n[RUN] {' '.join(map(str, cmd))}\n")
    return subprocess.run(
        cmd,
        check=True,
        cwd=cwd,
        text=True,
        capture_output=capture_output
    )


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"No se encontrÃ³ el archivo: {path}")


def ensure_vcf_index(vcf_file):
    tbi = Path(str(vcf_file) + ".tbi")
    csi = Path(str(vcf_file) + ".csi")
    if tbi.exists() or csi.exists():
        print(f"[OK] Ãndice VCF ya existe para: {vcf_file}")
        return
    run_cmd(["bcftools", "index", "-t", str(vcf_file)])


def get_sample_ids_from_vcf(vcf_file):
    res = run_cmd(["bcftools", "query", "-l", str(vcf_file)], capture_output=True)
    sample_ids = [x.strip() for x in res.stdout.splitlines() if x.strip()]
    if not sample_ids:
        raise RuntimeError(f"No pude extraer muestras desde el VCF: {vcf_file}")
    return sample_ids


def get_contig_names_from_vcf(vcf_file):
    ensure_vcf_index(vcf_file)
    res = run_cmd(["bcftools", "index", "-s", str(vcf_file)], capture_output=True)
    contigs = []
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        contig = line.split("\t")[0].strip()
        if contig:
            contigs.append(contig)
    if not contigs:
        raise RuntimeError(f"No pude extraer contigs desde el VCF: {vcf_file}")
    return contigs


def write_ploidy_file(vcf_file, outfile, ploidy_value=1):
    sample_ids = get_sample_ids_from_vcf(vcf_file)
    contigs = get_contig_names_from_vcf(vcf_file)

    with open(outfile, "w", encoding="utf-8") as out:
        out.write("#Inds\t" + "\t".join(contigs) + "\n")
        for sid in sample_ids:
            out.write(sid + "\t" + "\t".join([str(ploidy_value)] * len(contigs)) + "\n")
    print(f"[OK] Archivo de ploidÃ­a creado: {outfile}")


def summarize_diem_object(dobj, label):
    print(f"\n=== RESUMEN {label} ===")
    print(f"[INFO] nIndividuals: {len(dobj.indNames)}")
    print(f"[INFO] nChromosomes: {len(dobj.DMBC)}")

    sizes = []
    for i, chrom in enumerate(dobj.DMBC):
        try:
            n_sites = len(chrom)
        except Exception:
            n_sites = None
        sizes.append(n_sites)
        print(f"[INFO] chr_idx={i} n_sites={n_sites}")

    nonempty = [x for x in sizes if x not in (None, 0)]
    print(f"[INFO] nonempty chromosomes: {len(nonempty)}")

    if len(dobj.DMBC) == 0 or len(nonempty) == 0:
        raise RuntimeError(
            f"{label} quedÃ³ sin cromosomas/sitios utilizables. "
            "Si esto ocurre con META_CORRECTED, el problema estÃ¡ en update_meta/ploidy file. "
            "Si tambiÃ©n ocurre con META_RAW, el problema estÃ¡ antes: vcf2diem o el VCF."
        )


def save_current(outfile):
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close("all")


def sanitize_name(x):
    return str(x).replace("/", "_").replace(" ", "_").replace(":", "_")


def save_hi_plot(dobj, outfile, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dobj.HIs, marker=".", linestyle="none")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hybrid Index")
    ax.set_xlabel("Individuals ordered by HI")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def save_di_hist(dobj, outfile):
    di_vals = np.hstack(dobj.DIByChr)
    di_unique = np.unique(di_vals)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(di_vals, bins=500, alpha=0.7, label="all DI values")
    ax.hist(di_unique, bins=500, alpha=0.7, label="unique DI values")
    ax.set_yscale("log")
    ax.set_xlabel("Diagnostic Index (DI)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("DI distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def save_scale_test_plot(scales, changes, outfile):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scales, changes, marker=".")
    ax.set_xscale("log")
    ax.set_xlabel("Laplace smoothing scale")
    ax.set_ylabel("# sites changed")
    ax.set_title("Sensitivity to smoothing scale")
    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def load_diemtype(path):
    if hasattr(diem, "load_DiemType"):
        return diem.load_DiemType(str(path))
    raise AttributeError("Tu versiÃ³n de diempy no tiene load_DiemType().")


def get_chr_names(dobj):
    for attr in ["chromosomeNames", "chromNames", "chrNames", "contigNames"]:
        if hasattr(dobj, attr):
            vals = getattr(dobj, attr)
            if vals is not None and len(vals) == len(dobj.DMBC):
                return [str(v) for v in vals]
    return [f"contig{i}" for i in range(len(dobj.DMBC))]


def plot_hi(dobj, outfile, title, color=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    if color is None:
        ax.plot(dobj.HIs, marker=".")
    else:
        ax.plot(dobj.HIs, marker=".", color=color)
    ax.set_ylim(0, 1)
    ax.set_ylabel("HI")
    ax.set_xlabel("individual idx")
    ax.set_title(title)
    save_current(outfile)


def try_plot_painting(dm, names, title, outfile):
    try:
        plt.figure(figsize=(12, 8))
        diem.plot_painting(dm, names=names)
        plt.title(title)
        save_current(outfile)
    except Exception as e:
        print(f"[WARN] plot_painting fallÃ³ en {outfile.name}: {e}")
        plt.close("all")


def try_plot_painting_with_positions(dm, pos, names, title, outfile):
    try:
        plt.figure(figsize=(12, 8))
        diem.plot_painting_with_positions(dm, pos, names=names)
        plt.title(title)
        save_current(outfile)
    except Exception as e:
        print(f"[WARN] plot_painting_with_positions fallÃ³ en {outfile.name}: {e}")
        plt.close("all")


def export_all_intervals(dSmoothedSorted, chr_names, outfile):
    rows = []
    n_chr = dSmoothedSorted.contigMatrix.shape[0]
    n_ind = dSmoothedSorted.contigMatrix.shape[1]

    for chr_idx in range(n_chr):
        for ind_idx in range(n_ind):
            contig = dSmoothedSorted.contigMatrix[chr_idx, ind_idx]

            for interval_idx, interval in enumerate(contig.intervals):
                row = {
                    "chr_index": chr_idx,
                    "chr_name": chr_names[chr_idx] if chr_idx < len(chr_names) else f"contig{chr_idx}",
                    "ind_index": ind_idx,
                    "individual": dSmoothedSorted.indNames[ind_idx],
                    "interval_idx": interval_idx,
                    "state": getattr(interval, "state", np.nan),
                    "left_idx": getattr(interval, "left_idx", np.nan),
                    "right_idx": getattr(interval, "right_idx", np.nan),
                    "left_pos": getattr(interval, "left_pos", np.nan),
                    "right_pos": getattr(interval, "right_pos", np.nan),
                }

                try:
                    row["span"] = interval.span()
                except Exception:
                    row["span"] = np.nan

                try:
                    row["mapSpan"] = interval.mapSpan(dSmoothedSorted.chrLengths[chr_idx])
                except Exception:
                    row["mapSpan"] = np.nan

                rows.append(row)

    pd.DataFrame(rows).to_csv(outfile, sep="\t", index=False)


def detect_intermediate_indices(dobj):
    if USE_MANUAL_INTERMEDIATES:
        inds = MANUAL_INTERMEDIATES
    else:
        inds = [
            ind for ind, hi in zip(dobj.indNames, dobj.HIs)
            if INTERMEDIATE_HI_MIN <= hi <= INTERMEDIATE_HI_MAX
        ]

    name_to_idx = {name: i for i, name in enumerate(dobj.indNames)}
    idxs = [name_to_idx[x] for x in inds if x in name_to_idx]

    rows = []
    for ind in inds:
        if ind in name_to_idx:
            i = name_to_idx[ind]
            rows.append({"individual": ind, "idx": i, "HI": dobj.HIs[i]})
        else:
            rows.append({"individual": ind, "idx": np.nan, "HI": np.nan})

    return idxs, pd.DataFrame(rows)


def choose_smoothing_scale(df_or_path):
    if isinstance(df_or_path, (str, Path)):
        df = pd.read_csv(df_or_path, sep="\t").copy()
    else:
        df = df_or_path.copy()

    df = df.sort_values("scale").reset_index(drop=True)

    rel_drop = [np.nan]
    for i in range(1, df.shape[0]):
        prev = df.loc[i - 1, "changed_sites"]
        curr = df.loc[i, "changed_sites"]
        if prev == 0:
            rel_drop.append(0.0)
        else:
            rel_drop.append(abs(curr - prev) / abs(prev))

    df["relative_change_vs_prev"] = rel_drop

    candidates = df[
        (df["scale"] >= SMOOTH_MIN_SCALE) &
        (df["relative_change_vs_prev"] <= SMOOTH_RELATIVE_DELTA_THRESHOLD)
    ]

    if candidates.shape[0] > 0:
        chosen = float(candidates.iloc[0]["scale"])
        reason = "first_plateau"
    else:
        candidates = df[df["scale"] >= SMOOTH_MIN_SCALE].copy()
        if candidates.shape[0] == 0:
            candidates = df.copy()
        idx = candidates["relative_change_vs_prev"].replace(np.nan, np.inf).idxmin()
        chosen = float(candidates.loc[idx, "scale"])
        reason = "min_relative_change_fallback"

    return chosen, df, reason


def choose_di_threshold_from_table_or_fallback(path, fallback):
    if not path.exists():
        return fallback, None, "fallback_manual"

    df = pd.read_csv(path, sep="\t").copy()

    required = {"threshold", "min_HI", "max_HI"}
    if not required.issubset(df.columns):
        return fallback, df, "fallback_table_missing_columns"

    if "retained_fraction" not in df.columns:
        df["retained_fraction"] = 1.0

    df = df.sort_values("threshold").reset_index(drop=True)

    touchdown = df[
        (df["min_HI"] <= TOUCHDOWN_TOL) &
        (df["max_HI"] >= 1 - TOUCHDOWN_TOL) &
        (df["retained_fraction"] >= MIN_RETAINED_FRACTION)
    ].copy()

    if touchdown.shape[0] > 0:
        chosen = float(touchdown.iloc[0]["threshold"])
        return chosen, df, "touchdown_from_table"

    return fallback, df, "fallback_no_touchdown_match"


def choose_left_right_from_hi_table(path):
    require_file(path)
    df = pd.read_csv(path, sep="\t").copy()

    hi_col = None
    for c in df.columns:
        if c.startswith("HI"):
            hi_col = c
            break
    if hi_col is None:
        raise ValueError(f"No encontrÃ© columna HI en {path}")

    df = df.sort_values(hi_col).reset_index(drop=True)
    his = df[hi_col].to_numpy()

    if len(his) < 2:
        raise ValueError("No hay suficientes individuos para definir left/right")

    gaps = np.diff(his)

    valid_start = MIN_EDGE_BUFFER - 1
    valid_end = len(gaps) - MIN_EDGE_BUFFER + 1

    if valid_end <= valid_start:
        split_idx = int(np.argmax(gaps))
        reason = "largest_gap_no_buffer"
    else:
        subgaps = gaps[valid_start:valid_end]
        split_idx = int(np.argmax(subgaps) + valid_start)
        reason = "largest_internal_gap"

    left_idx = np.arange(split_idx + 1)
    right_idx = np.arange(split_idx + 1, len(his))

    return left_idx, right_idx, df, hi_col, reason, float(gaps[split_idx])


# =========================================================
# 0. SUBSET DEL VCF
# =========================================================

print("\n=== 0. Subset del VCF ===")
print(f"[INFO] SOURCE_VCF: {SOURCE_VCF}")
print(f"[INFO] POPMAP: {POPMAP}")
print(f"[INFO] OUTDIR: {OUTDIR}")

require_file(SOURCE_VCF)
require_file(POPMAP)

if RUN_SUBSET:
    pop_expr = " || ".join([f'$2==\"{p}\"' for p in POPULATIONS])
    awk_cmd = f"awk '{pop_expr} {{print $1}}' \"{POPMAP}\" > \"{SAMPLES_FILE}\""
    run_cmd(["bash", "-lc", awk_cmd], cwd=str(RUN_DIR))

    require_file(SAMPLES_FILE)

    print("[INFO] Muestras seleccionadas:")
    with open(SAMPLES_FILE, "r", encoding="utf-8") as fh:
        samples = [x.strip() for x in fh if x.strip()]
    for s in samples:
        print(s)
    print(f"[INFO] n_samples = {len(samples)}")

    run_cmd([
        "bcftools", "view",
        "-S", str(SAMPLES_FILE),
        "-Oz",
        "-o", str(VCF),
        str(SOURCE_VCF)
    ], cwd=str(RUN_DIR))

    run_cmd(["bcftools", "index", "-t", str(VCF)], cwd=str(RUN_DIR))
else:
    require_file(VCF)

print(f"[OK] VCF subset generado: {VCF}")


# =========================================================
# 1. PREPARACIÃ“N
# =========================================================

print("\n=== 1. PreparaciÃ³n ===")
print(f"[INFO] RUN_DIR = {RUN_DIR}")
print(f"[INFO] VCF = {VCF}")

require_file(VCF)
ensure_vcf_index(VCF)

if RUN_VCF2DIEM:
    print("\n=== Ejecutando vcf2diem ===")
    run_cmd(["vcf2diem", str(VCF)], cwd=str(RUN_DIR))
else:
    print("\n=== Saltando vcf2diem; se usarÃ¡n archivos ya existentes ===")

require_file(META_RAW)
require_file(BED_IN)

print(f"[OK] META_RAW encontrado: {META_RAW}")
print(f"[OK] BED_IN encontrado: {BED_IN}")


# =========================================================
# 2. ARCHIVO DE PLOIDÃA
# =========================================================
# Fiel al tutorial haploide: solo corregimos ploidÃ­a.
# No usamos recFile porque todas las tasas relativas son 1.0.

print("\n=== 2. Archivo de ploidÃ­a ===")
write_ploidy_file(VCF, PLOIDY_FILE, ploidy_value=1)


# =========================================================
# 3. ACTUALIZAR META
# =========================================================

print("\n=== 3. update_meta ===")
diem.update_meta(
    str(META_RAW),
    str(META_CORRECTED),
    ploidyFilePath=str(PLOIDY_FILE)
)

require_file(META_CORRECTED)
print(f"[OK] META_CORRECTED generado: {META_CORRECTED}")


# =========================================================
# 4. LECTURA Y DIAGNÃ“STICO
# =========================================================

print("\n=== 4. Lectura de datos ===")

dRaw_from_rawmeta = diem.read_diem_bed(str(BED_IN), str(META_RAW))
summarize_diem_object(dRaw_from_rawmeta, "dRaw_from_rawmeta")

dRaw = diem.read_diem_bed(str(BED_IN), str(META_CORRECTED))
summarize_diem_object(dRaw, "dRaw_from_corrected_meta")


# =========================================================
# 5. POLARIZACIÃ“N
# =========================================================

print("\n=== 5. PolarizaciÃ³n ===")
dPol = dRaw.polarize(ncores=NCORES)
dPol.sort()

diem.write_polarized_bed(str(BED_IN), str(BED_POL), dPol)
diem.save_DiemType(dPol, str(DIEMTYPE_POL))

save_hi_plot(
    dPol,
    PLOT_DIR / f"{DATASET_TAG}.01_hybrid_index_polarized.png",
    f"{DATASET_TAG}: Hybrid Index after polarization"
)

save_di_hist(
    dPol,
    PLOT_DIR / f"{DATASET_TAG}.02_DI_distribution.png"
)

pd.DataFrame({
    "individual": dPol.indNames,
    "HI": dPol.HIs
}).to_csv(
    TABLES_IN / f"{DATASET_TAG}.hybrid_index_polarized.tsv",
    sep="\t",
    index=False
)


# =========================================================
# 5.1 ESCANEO DE THRESHOLDS DI
# =========================================================

print("\n=== 5.1. Escaneo de thresholds DI ===")
threshold_scan_rows = []

for thr in sorted(np.unique(np.concatenate([
    np.arange(-200, -9, 10),
    np.array([-8, -7, -6, -5, -4, -3, -2, -1])
]))):
    try:
        d_tmp = dPol.apply_threshold(float(thr))
        d_tmp.sort()

        hi_vals = np.array(d_tmp.HIs, dtype=float)

        retained_sites = 0
        total_sites = 0
        try:
            for chr_idx in range(len(d_tmp.DMBC)):
                try:
                    retained_sites += len(d_tmp.DMBC[chr_idx])
                except Exception:
                    pass
            for chr_idx in range(len(dPol.DMBC)):
                try:
                    total_sites += len(dPol.DMBC[chr_idx])
                except Exception:
                    pass
        except Exception:
            retained_sites = np.nan
            total_sites = np.nan

        retained_fraction = (
            retained_sites / total_sites
            if isinstance(retained_sites, (int, float)) and isinstance(total_sites, (int, float)) and total_sites not in [0, np.nan]
            else np.nan
        )

        threshold_scan_rows.append({
            "threshold": float(thr),
            "min_HI": float(np.min(hi_vals)) if len(hi_vals) > 0 else np.nan,
            "max_HI": float(np.max(hi_vals)) if len(hi_vals) > 0 else np.nan,
            "range_HI": float(np.max(hi_vals) - np.min(hi_vals)) if len(hi_vals) > 0 else np.nan,
            "n_individuals": len(hi_vals),
            "retained_sites": retained_sites,
            "total_sites": total_sites,
            "retained_fraction": retained_fraction
        })
    except Exception as e:
        threshold_scan_rows.append({
            "threshold": float(thr),
            "min_HI": np.nan,
            "max_HI": np.nan,
            "range_HI": np.nan,
            "n_individuals": np.nan,
            "retained_sites": np.nan,
            "total_sites": np.nan,
            "retained_fraction": np.nan,
            "error": str(e)
        })

df_threshold_scan = pd.DataFrame(threshold_scan_rows)
df_threshold_scan.to_csv(
    TABLES_IN / f"{DATASET_TAG}.DI_touchdown_scan.tsv",
    sep="\t",
    index=False
)


# =========================================================
# 6. THRESHOLD AUTOMÃTICO
# =========================================================

print("\n=== 6. Threshold DI automÃ¡tico ===")
DI_THRESHOLD_AUTO, df_di_scan, di_reason = choose_di_threshold_from_table_or_fallback(
    DI_TOUCHDOWN_TABLE,
    FALLBACK_DI_THRESHOLD
)

print(f"[AUTO] DI_THRESHOLD = {DI_THRESHOLD_AUTO} ({di_reason})")

if df_di_scan is not None:
    df_di_scan.to_csv(
        TABLE_DIR / f"{DATASET_TAG}.DI_touchdown_scan_copy.tsv",
        sep="\t",
        index=False
    )

pd.DataFrame([{
    "dataset_tag": DATASET_TAG,
    "di_threshold_auto": DI_THRESHOLD_AUTO,
    "di_threshold_reason": di_reason,
    "fallback_di_threshold": FALLBACK_DI_THRESHOLD,
    "populations": ",".join(POPULATIONS)
}]).to_csv(
    TABLE_DIR / f"{DATASET_TAG}.threshold_metadata.tsv",
    sep="\t",
    index=False
)

dThresh = dPol.apply_threshold(DI_THRESHOLD_AUTO)
dThresh.sort()

diem.save_DiemType(dThresh, str(DIEMTYPE_THRESH))

save_hi_plot(
    dThresh,
    PLOT_DIR / f"{DATASET_TAG}.03_hybrid_index_thresholded.png",
    f"{DATASET_TAG}: Hybrid Index after DI threshold ({DI_THRESHOLD_AUTO})"
)

pd.DataFrame({
    "individual": dThresh.indNames,
    "HI": dThresh.HIs
}).to_csv(
    TABLES_IN / f"{DATASET_TAG}.hybrid_index_thresholded.tsv",
    sep="\t",
    index=False
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dPol.HIs, marker=".", linestyle="none", label="polarized")
ax.plot(dThresh.HIs, marker=".", linestyle="none", label=f"thresholded {DI_THRESHOLD_AUTO}")
ax.set_ylim(0, 1)
ax.set_ylabel("Hybrid Index")
ax.set_xlabel("Individuals ordered by HI")
ax.set_title(f"{DATASET_TAG}: HI before vs after thresholding")
ax.legend()
plt.tight_layout()
fig.savefig(PLOT_DIR / f"{DATASET_TAG}.04_HI_before_vs_after_threshold.png", dpi=300)
plt.close(fig)


# =========================================================
# 7. SMOOTHING TEST
# =========================================================

print("\n=== 7. Smoothing test ===")
sitesDiffByScale = []

for scale in SMOOTH_SCALES:
    try:
        dSmoothedTest = dThresh.smooth(scale)
        kdiffs = diem.count_site_differences(dSmoothedTest.DMBC, dThresh.DMBC)
        sitesDiffByScale.append(kdiffs)
        print(f"[SMOOTH TEST] scale={scale:.2e} changed_sites={kdiffs}")
    except Exception as e:
        print(f"[WARN] FallÃ³ smooth({scale}): {e}")
        sitesDiffByScale.append(np.nan)

df_smooth = pd.DataFrame({
    "scale": SMOOTH_SCALES,
    "changed_sites": sitesDiffByScale
})

if AUTO_SELECT_SMOOTH:
    FINAL_SMOOTH_SCALE_AUTO, df_smooth_eval, smooth_reason = choose_smoothing_scale(df_smooth)
else:
    FINAL_SMOOTH_SCALE_AUTO = FINAL_SMOOTH_SCALE
    df_smooth_eval = df_smooth.copy()
    smooth_reason = "manual"

df_smooth_eval.to_csv(
    TABLES_IN / f"{DATASET_TAG}.smoothing_scale_sensitivity.tsv",
    sep="\t",
    index=False
)

pd.DataFrame([{
    "dataset_tag": DATASET_TAG,
    "final_smooth_scale_auto": FINAL_SMOOTH_SCALE_AUTO,
    "smooth_reason": smooth_reason,
    "fallback_final_smooth_scale": FINAL_SMOOTH_SCALE
}]).to_csv(
    TABLE_DIR / f"{DATASET_TAG}.smoothing_metadata.tsv",
    sep="\t",
    index=False
)

save_scale_test_plot(
    SMOOTH_SCALES,
    sitesDiffByScale,
    PLOT_DIR / f"{DATASET_TAG}.05_smoothing_sensitivity.png"
)


# =========================================================
# 8. SMOOTHING FINAL
# =========================================================

print("\n=== 8. Smoothing final ===")
print(f"[AUTO] FINAL_SMOOTH_SCALE = {FINAL_SMOOTH_SCALE_AUTO} ({smooth_reason})")

dSmoothedUnsorted = dThresh.smooth(FINAL_SMOOTH_SCALE_AUTO)
dSmoothed = dSmoothedUnsorted.copy()
dSmoothed.sort()

diem.save_DiemType(dSmoothed, str(DIEMTYPE_SMOOTH))

save_hi_plot(
    dSmoothed,
    PLOT_DIR / f"{DATASET_TAG}.06_hybrid_index_smoothed.png",
    f"{DATASET_TAG}: Hybrid Index after smoothing ({FINAL_SMOOTH_SCALE_AUTO})"
)

pd.DataFrame({
    "individual": dSmoothed.indNames,
    "HI": dSmoothed.HIs
}).to_csv(
    TABLES_IN / f"{DATASET_TAG}.hybrid_index_smoothed.tsv",
    sep="\t",
    index=False
)


# =========================================================
# 9. CONTIGS
# =========================================================

print("\n=== 9. Contigs ===")
dSmoothed.create_contig_matrix()
diem.export_contigs_to_ind_bed_files(dSmoothed, str(CONTIG_DIR))


# =========================================================
# 10. TRACTOS ESTADO 1
# =========================================================

print("\n=== 10. Tractos estado 1 ===")
oneTracts = dSmoothed.get_intervals_of_state(1)
oneLengths = [x.span() for x in oneTracts]

pd.DataFrame({
    "tract_length_state1": oneLengths
}).to_csv(
    TABLES_IN / f"{DATASET_TAG}.tract_lengths_state1.tsv",
    sep="\t",
    index=False
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(oneLengths, bins=50)
ax.set_xlabel("Tract length")
ax.set_ylabel("Count")
ax.set_title(f"{DATASET_TAG}: Distribution of tract lengths (state 1)")
plt.tight_layout()
fig.savefig(PLOT_DIR / f"{DATASET_TAG}.07_tract_lengths_state1.png", dpi=300)
plt.close(fig)


# =========================================================
# 11. RECARGAR DIEMTYPES
# =========================================================

print("\n=== 11. Recargando diemtypes ===")
require_file(DIEMTYPE_POL)
require_file(DIEMTYPE_THRESH)
require_file(DIEMTYPE_SMOOTH)
require_file(BED_POL)
require_file(META_CORRECTED)

dPol = load_diemtype(DIEMTYPE_POL)
dThresh = load_diemtype(DIEMTYPE_THRESH)
dSmoothed = load_diemtype(DIEMTYPE_SMOOTH)

dPol.sort()
dThresh.sort()
dSmoothed.sort()

chr_names = get_chr_names(dPol)
n_chr = len(dPol.DMBC)

if CONTIGS_TO_PLOT is None:
    CONTIGS_TO_PLOT = list(range(n_chr))
if MULTI_CHR_LIST is None:
    MULTI_CHR_LIST = list(range(n_chr))


# =========================================================
# 12. TABLAS HI Y LEFT/RIGHT
# =========================================================

print("\n=== 12. Tablas HI ===")
pol_hi_path = TABLE_DIR / "dPol_HI.tsv"
thresh_hi_path = TABLE_DIR / "dThresh_HI.tsv"
smoothed_hi_path = TABLE_DIR / "dSmoothed_HI.tsv"

pd.DataFrame({
    "idx": np.arange(len(dPol.indNames)),
    "individual": dPol.indNames,
    "HI_dPol": dPol.HIs
}).to_csv(pol_hi_path, sep="\t", index=False)

pd.DataFrame({
    "idx": np.arange(len(dThresh.indNames)),
    "individual": dThresh.indNames,
    "HI_dThresh": dThresh.HIs
}).to_csv(thresh_hi_path, sep="\t", index=False)

pd.DataFrame({
    "idx": np.arange(len(dSmoothed.indNames)),
    "individual": dSmoothed.indNames,
    "HI_dSmoothed": dSmoothed.HIs
}).to_csv(smoothed_hi_path, sep="\t", index=False)

if AUTO_SELECT_LEFT_RIGHT:
    left_idx, right_idx, hi_df, hi_col, lr_reason, gap_val = choose_left_right_from_hi_table(smoothed_hi_path)

    pd.DataFrame({
        "group": ["left"] * len(left_idx) + ["right"] * len(right_idx),
        "idx": list(left_idx) + list(right_idx),
        "individual": [dSmoothed.indNames[i] for i in list(left_idx) + list(right_idx)],
        "HI": [dSmoothed.HIs[i] for i in list(left_idx) + list(right_idx)]
    }).to_csv(
        TABLE_DIR / "left_right_indices_used.tsv",
        sep="\t",
        index=False
    )

    pd.DataFrame([{
        "reason": lr_reason,
        "gap_value": gap_val,
        "hi_column": hi_col
    }]).to_csv(
        TABLE_DIR / "left_right_metadata.tsv",
        sep="\t",
        index=False
    )
else:
    left_idx = None
    right_idx = None

plot_hi(
    dPol,
    SUMMARY_DIR / "01_dPol_HI_sorted.png",
    f"{DATASET_TAG} - dPol HI after sort"
)


# =========================================================
# 13. PAINTINGS POR CONTIG
# =========================================================

print("\n=== 13. Paintings por contig ===")
for i in CONTIGS_TO_PLOT:
    if i >= n_chr:
        continue

    chr_name = chr_names[i]
    chr_tag = sanitize_name(chr_name)

    try_plot_painting(
        dPol.DMBC[i],
        dPol.indNames,
        f"{DATASET_TAG} - dPol painting - {chr_name}",
        PAINT_DIR / f"01_dPol_painting_chr{i:02d}_{chr_tag}.png"
    )

    try_plot_painting_with_positions(
        dPol.DMBC[i],
        dPol.posByChr[i],
        dPol.indNames,
        f"{DATASET_TAG} - dPol with positions - {chr_name}",
        PAINT_DIR / f"02_dPol_with_positions_chr{i:02d}_{chr_tag}.png"
    )

    try_plot_painting_with_positions(
        dThresh.DMBC[i],
        dThresh.posByChr[i],
        dThresh.indNames,
        f"{DATASET_TAG} - dThresh with positions - {chr_name}",
        PAINT_DIR / f"03_dThresh_with_positions_chr{i:02d}_{chr_tag}.png"
    )

    try_plot_painting(
        dThresh.DMBC[i],
        dThresh.indNames,
        f"{DATASET_TAG} - dThresh painting - {chr_name}",
        PAINT_DIR / f"04_dThresh_painting_chr{i:02d}_{chr_tag}.png"
    )

    try_plot_painting(
        dSmoothed.DMBC[i],
        dSmoothed.indNames,
        f"{DATASET_TAG} - dSmoothed painting - {chr_name}",
        PAINT_DIR / f"05_dSmoothed_painting_chr{i:02d}_{chr_tag}.png"
    )


# =========================================================
# 14. DI DISTRIBUTION Y COMPARACIONES DE HI
# =========================================================

print("\n=== 14. DI distribution y comparaciones de HI ===")
DIValsWithRepeats = np.hstack(dPol.DIByChr)
DIValsUnique = np.unique(DIValsWithRepeats)

fig, ax = plt.subplots(figsize=(5, 4))
counts, bins, patches = ax.hist(DIValsWithRepeats, bins=500)
ax.hist(DIValsUnique, bins=bins, alpha=0.7)
plt.yscale("log")
ax.set_xlabel("DI")
ax.set_ylabel("count")
ax.set_title(f"{DATASET_TAG} - DI distribution")
save_current(SUMMARY_DIR / "02_DI_distribution.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(dPol.HIs, marker=".", label="dPol")
ax.plot(dThresh.HIs, marker=".", label="dThresh")
ax.set_ylim(0, 1)
ax.set_ylabel("HI")
ax.set_xlabel("individual idx")
ax.set_title(f"{DATASET_TAG} - HI before vs after threshold")
ax.legend()
save_current(SUMMARY_DIR / "03_HI_before_vs_after_threshold.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(SMOOTH_SCALES, sitesDiffByScale, marker=".")
ax.set_xscale("log")
ax.axvline(FINAL_SMOOTH_SCALE_AUTO, linestyle="--")
ax.set_ylabel("# of sites changed")
ax.set_xlabel("Laplace scale")
ax.set_title(f"{DATASET_TAG} - smoothing sensitivity")
save_current(SUMMARY_DIR / "04_smoothing_sensitivity.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(dThresh.HIs, marker=".", label="dThresh")
ax.plot(dSmoothed.computeHIs(), marker=".", label="smoothed recomputed")
ax.set_ylabel("HI")
ax.set_xlabel("individual idx")
ax.set_title(f"{DATASET_TAG} - HI before vs after smoothing")
ax.legend()
save_current(SUMMARY_DIR / "05_HI_before_vs_after_smoothing.png")


# =========================================================
# 15. CONTIG MATRIX E INTERVALOS
# =========================================================

print("\n=== 15. Contig matrix e intervalos ===")
try:
    dSmoothed.create_contig_matrix()
except Exception:
    pass

contig_summary_rows = []
for chr_idx in range(dSmoothed.contigMatrix.shape[0]):
    for ind_idx in range(dSmoothed.contigMatrix.shape[1]):
        contig = dSmoothed.contigMatrix[chr_idx, ind_idx]
        contig_summary_rows.append({
            "chr_index": chr_idx,
            "chr_name": chr_names[chr_idx] if chr_idx < len(chr_names) else f"contig{chr_idx}",
            "ind_index": ind_idx,
            "individual": dSmoothed.indNames[ind_idx],
            "num_intervals": contig.num_intervals
        })

pd.DataFrame(contig_summary_rows).to_csv(
    TABLE_DIR / "all_contigs_summary.tsv",
    sep="\t",
    index=False
)

export_all_intervals(
    dSmoothed,
    chr_names,
    TABLE_DIR / "all_contigs_all_intervals.tsv"
)


# =========================================================
# 16. TRACTOS POR ESTADO Y POR CONTIG
# =========================================================

print("\n=== 16. Tractos por estado y por contig ===")
for state in [1, 2, 3]:
    try:
        spans = [x.span() for x in dSmoothed.get_intervals_of_state(state)]
        if len(spans) == 0:
            continue
        fig, ax = plt.subplots()
        ax.hist(spans, bins=50)
        ax.set_ylabel("counts")
        ax.set_xlabel("tract length")
        ax.set_title(f"{DATASET_TAG} - state {state} tract lengths")
        save_current(TRACT_DIR / f"01_state{state}_tract_lengths.png")
    except Exception as e:
        print(f"[WARN] tract plot global fallÃ³ para state {state}: {e}")

fig, ax = plt.subplots()
ok_any = False
for state, alpha in zip([1, 2, 3], [0.5, 0.5, 0.5]):
    try:
        spans = [x.span() for x in dSmoothed.get_intervals_of_state(state)]
        if len(spans) > 0:
            ax.hist(spans, bins=50, alpha=alpha, label=f"state {state}")
            ok_any = True
    except Exception:
        pass

if ok_any:
    ax.set_ylabel("counts")
    ax.set_xlabel("tract length")
    ax.set_title(f"{DATASET_TAG} - tract lengths by state")
    ax.legend()
    save_current(TRACT_DIR / "02_states123_tract_lengths.png")
else:
    plt.close("all")

for chr_idx, chr_name in enumerate(chr_names):
    chr_tag = sanitize_name(chr_name)
    for state in [1, 2, 3]:
        try:
            spans = [x.span() for x in dSmoothed.get_intervals_of_state(state, chromosomeSubset=[chr_idx])]
            if len(spans) == 0:
                continue

            fig, ax = plt.subplots()
            ax.hist(spans, bins=50)
            ax.set_ylabel("counts")
            ax.set_xlabel("tract length")
            ax.set_title(f"{DATASET_TAG} - state {state} tract lengths - {chr_name}")
            save_current(TRACT_DIR / f"03_state{state}_tract_lengths_chr{chr_idx:02d}_{chr_tag}.png")
        except Exception as e:
            print(f"[WARN] tract plot fallÃ³ para {chr_name}, state {state}: {e}")


# =========================================================
# 17. WHOLE-GENOME PLOTS
# =========================================================

print("\n=== 17. Whole-genome plots ===")
try:
    plotprep = diem.diemPlotPrepFromBedMeta(
        plot_theme=DATASET_TAG,
        bed_file_path=str(BED_POL),
        meta_file_path=str(META_CORRECTED),
        di_threshold=DI_THRESHOLD_AUTO,
        genome_pixels=GENOME_PIXELS,
        ticks=TICKS,
        smooth=FINAL_SMOOTH_SCALE_AUTO
    )

    diem.diemIrisFromPlotPrep(plotprep)
    save_current(WHOLEGENOME_DIR / f"{DATASET_TAG}.iris.circular.png")

    diem.diemLongFromPlotPrep(plotprep, list(range(n_chr)))
    save_current(WHOLEGENOME_DIR / f"{DATASET_TAG}.long.rectangular_all_contigs.png")

    for i in range(n_chr):
        diem.diemLongFromPlotPrep(plotprep, [i])
        save_current(WHOLEGENOME_DIR / f"{DATASET_TAG}.long.rectangular_contig_{i:02d}.png")

except Exception as e:
    print(f"[WARN] Whole-genome plots fallaron: {e}")


# =========================================================
# 18. CONTIG AND INDIVIDUAL CONTRIBUTIONS
# =========================================================

print("\n=== 18. Contig and individual contributions ===")
try:
    diem.GenomicContributionsPlot(dSmoothed, range(len(dSmoothed.DMBC)))
    save_current(CONTIG_CONTRIB_DIR / "genomic_contributions_all_contigs.png")
except Exception as e:
    print(f"[WARN] GenomicContributionsPlot global fallÃ³: {e}")

for start in range(0, len(dSmoothed.DMBC), CONTIG_CHUNK_SIZE):
    stop = min(start + CONTIG_CHUNK_SIZE, len(dSmoothed.DMBC))
    idxs = range(start, stop)
    try:
        diem.GenomicContributionsPlot(dSmoothed, idxs)
        save_current(CONTIG_CONTRIB_DIR / f"genomic_contributions_contigs_{start:02d}_{stop-1:02d}.png")
    except Exception as e:
        print(f"[WARN] GenomicContributionsPlot {start}-{stop-1} fallÃ³: {e}")

intermediate_idxs, df_inter = detect_intermediate_indices(dSmoothed)
df_inter.to_csv(TABLE_DIR / "intermediate_individuals.tsv", sep="\t", index=False)

if len(intermediate_idxs) > 0:
    try:
        diem.IndGenomicContributionsPlot(dSmoothed, intermediate_idxs)
        save_current(IND_CONTRIB_DIR / "individual_genomic_contributions_intermediates.png")
    except Exception as e:
        print(f"[WARN] IndGenomicContributionsPlot conjunto fallÃ³: {e}")

    for idx in intermediate_idxs:
        ind_name = dSmoothed.indNames[idx]
        ind_tag = sanitize_name(ind_name)
        try:
            diem.IndGenomicContributionsPlot(dSmoothed, [idx])
            save_current(IND_CONTRIB_DIR / f"individual_contribution_{idx:03d}_{ind_tag}.png")
        except Exception as e:
            print(f"[WARN] IndGenomicContributionsPlot individual fallÃ³ para {ind_name}: {e}")


# =========================================================
# 19. INTERACTIVE-LIKE PLOTS
# =========================================================

print("\n=== 19. Interactive-like plots ===")
interactive_calls = [
    ("GenomeSummaryPlot", lambda: diem.GenomeSummaryPlot(dPol)),
    ("GenomicDeFinettiPlot", lambda: diem.GenomicDeFinettiPlot(dPol)),
    ("GenomicContributionsPlot", lambda: diem.GenomicContributionsPlot(dPol, range(len(dPol.DMBC)))),
]

for name, func in interactive_calls:
    try:
        func()
        save_current(INTERACTIVE_DIR / f"{name}.png")
    except Exception as e:
        print(f"[WARN] {name} no se pudo guardar/ejecutar: {e}")

try:
    diem.GenomeMultiSummaryPlot(dPol, MULTI_CHR_LIST)
    save_current(INTERACTIVE_DIR / "GenomeMultiSummaryPlot.png")
except Exception as e:
    print(f"[WARN] GenomeMultiSummaryPlot fallÃ³: {e}")

try:
    diem.GenomicMultiDeFinettiPlot(dPol, MULTI_CHR_LIST)
    save_current(INTERACTIVE_DIR / "GenomicMultiDeFinettiPlot.png")
except Exception as e:
    print(f"[WARN] GenomicMultiDeFinettiPlot fallÃ³: {e}")


# =========================================================
# 20. RESUMEN FINAL
# =========================================================

print("\n=== 20. Resumen final ===")
df_hi_sorted = pd.DataFrame({
    "idx": np.arange(len(dSmoothed.indNames)),
    "individual": dSmoothed.indNames,
    "HI": dSmoothed.HIs
}).sort_values("HI").reset_index(drop=True)

df_hi_sorted.to_csv(TABLE_DIR / "individuals_HI_sorted.tsv", sep="\t", index=False)

plt.figure(figsize=(7, 4))
plt.hist(dSmoothed.HIs, bins=20)
plt.xlabel("Hybrid Index")
plt.ylabel("Count")
plt.title(f"Distribution of Hybrid Index in {DATASET_TAG}")
save_current(TABLE_DIR / "HI_distribution.png")

pd.DataFrame([{
    "dataset_tag": DATASET_TAG,
    "source_vcf": str(SOURCE_VCF),
    "subset_vcf": str(VCF),
    "populations": ",".join(POPULATIONS),
    "n_individuals_final": len(dSmoothed.indNames),
    "di_threshold_auto": DI_THRESHOLD_AUTO,
    "di_threshold_reason": di_reason,
    "smooth_scale_auto": FINAL_SMOOTH_SCALE_AUTO,
    "smooth_reason": smooth_reason,
    "n_contigs": n_chr
}]).to_csv(
    TABLE_DIR / "run_metadata.tsv",
    sep="\t",
    index=False
)

print("\n=== FIN ===")
print(f"Outputs bÃ¡sicos en: {RUN_DIR}")
print(f"Outputs extendidos en: {OUTPLOT_DIR}")
print(f"Contigs procesados: {n_chr}")
