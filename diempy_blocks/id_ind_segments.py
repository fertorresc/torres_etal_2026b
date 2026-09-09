#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llamada reproducible de bloques ancestry-discordant desde objetos diemPy
thresholded, sin kernel smoothing y sin HMM.

Características:
- Descubre automáticamente todos los *.thresholded.diemtype del directorio.
- Usa referencias biológicas explícitas y obligatorias definidas en el runner; no las infiere desde los datos ni desde el nombre.
- Asocia cada objeto thresholded con su polarized y PopMap.
- Detecta automáticamente haploides/diploides desde nombre, PopMap y estados.
- Divide poblaciones focales bimodales (p. ej. ZTCN) por background individual.
- Haploides: llama haplotipos donor-like.
- Diploides: separa bloques primarios con dos copias donor-like y tramos de
  soporte heterocigoto con una copia donor-like.
- Conserva los estados originales; no suaviza ni rellena SNPs.
- Registra coordenadas, tamaño, tamaño del contig, proporción del contig,
  dirección, portadores, densidad de soporte y controles de coherencia.
- Consolida bloques individuales por complete-link.
- Compara automáticamente replicación haploide/diploide.
- Todos los archivos llevan en el nombre los parámetros principales.
- Imprime un resumen completo en terminal al finalizar.

Los límites son el primer y último SNP de soporte; no son breakpoints exactos.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
import pickle
import platform
import re
import shutil
import sys
import traceback
from collections import defaultdict
from fnmatch import fnmatchcase
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

VERSION = "2026-07-20-fixed-references-v4.0"


# -----------------------------------------------------------------------------
# General
# -----------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(text: Any) -> str:
    s = str(text).strip()
    s = s.replace("-", "minus") if re.fullmatch(r"-?\d+(?:\.\d+)?", s) else s
    s = re.sub(r"[^A-Za-z0-9._]+", "_", s)
    return s.strip("_") or "NA"


def num_tag(x: float) -> str:
    if float(x).is_integer():
        raw = str(int(x))
    else:
        raw = f"{float(x):g}"
    return raw.replace("-", "minus").replace(".", "p")


def fraction_tag(x: float) -> str:
    return num_tag(float(x))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.name.endswith(".gz") else None
    df.to_csv(path, sep="\t", index=False, compression=compression)


def normalize_pop(value: str) -> str:
    pop = str(value).strip().upper()
    pop = re.sub(r"(?:[_\-. ]+(?:HAP|HAPLOID|DIP|DIPLOID|DIPLOIDE))$", "", pop, flags=re.I)
    if pop == "CONS":
        return "CON"
    return pop


def parse_popmap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", dtype=str, engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"PopMap inválido: {path}")
    df = df.iloc[:, :2].copy()
    df.columns = ["sample", "group"]
    df["sample"] = df["sample"].str.strip()
    df["group"] = df["group"].str.strip()
    df["population"] = df["group"].map(normalize_pop)
    if df["sample"].duplicated().any():
        dup = df.loc[df["sample"].duplicated(), "sample"].tolist()
        raise ValueError(f"PopMap con IDs duplicados: {dup[:10]}")
    return df


# -----------------------------------------------------------------------------
# DiemType loading
# -----------------------------------------------------------------------------

def _get_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _load_raw_diemtype(path: Path) -> Any:
    # Prefer the official loader when diempy is installed in the user's env.
    try:
        import diempy as diem  # type: ignore
        loader = getattr(diem, "load_DiemType", None)
        if loader is not None:
            return loader(str(path))
    except Exception:
        pass

    # Fallback for dictionary-serialized DiemType objects.
    with path.open("rb") as fh:
        return pickle.load(fh)


def canonicalize_diemtype(path: Path) -> Dict[str, Any]:
    raw = _load_raw_diemtype(path)
    out = {
        "DMBC": _get_attr(raw, ["DMBC"]),
        "indNames": _get_attr(raw, ["indNames", "individualNames"]),
        "chrNames": _get_attr(raw, ["chrNames", "chromosomeNames", "chromNames", "contigNames"]),
        "posByChr": _get_attr(raw, ["posByChr", "positionsByChr", "positionByChr"]),
        "chrLengths": _get_attr(raw, ["chrLengths", "chromosomeLengths", "contigLengths"]),
        "DIByChr": _get_attr(raw, ["DIByChr"]),
        "HIs": _get_attr(raw, ["HIs"]),
        "threshold": _get_attr(raw, ["threshold", "DIthreshold", "diThreshold"]),
        "smoothScale": _get_attr(raw, ["smoothScale", "smoothingScale"]),
        "contigMatrix": _get_attr(raw, ["contigMatrix"]),
    }
    required = ["DMBC", "indNames", "posByChr", "DIByChr"]
    missing = [k for k in required if out[k] is None]
    if missing:
        raise KeyError(f"DiemType incompleto {path}; faltan {missing}")

    n_chr = len(out["DMBC"])
    if out["chrNames"] is None:
        out["chrNames"] = [f"contig{i}" for i in range(n_chr)]
    if out["chrLengths"] is None:
        out["chrLengths"] = [int(np.max(np.asarray(p))) if len(p) else 0 for p in out["posByChr"]]
    if out["HIs"] is None:
        out["HIs"] = np.full(len(out["indNames"]), np.nan)

    out["indNames"] = [str(x) for x in out["indNames"]]
    out["chrNames"] = [str(x) for x in out["chrNames"]]
    out["chrLengths"] = [int(x) for x in out["chrLengths"]]
    return out


def orient_matrix(dm: np.ndarray, n_ind: int, n_sites: int, label: str) -> np.ndarray:
    dm = np.asarray(dm)
    if dm.ndim != 2:
        raise ValueError(f"DMBC no bidimensional en {label}: shape={dm.shape}")
    if dm.shape == (n_ind, n_sites):
        return dm
    if dm.shape == (n_sites, n_ind):
        return dm.T
    raise ValueError(f"DMBC incompatible en {label}: shape={dm.shape}, individuos={n_ind}, sitios={n_sites}")


def canonical_contig(obj: Dict[str, Any], ci: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(obj["posByChr"][ci], dtype=int)
    di = np.asarray(obj["DIByChr"][ci], dtype=float)
    dm = orient_matrix(np.asarray(obj["DMBC"][ci]), len(obj["indNames"]), len(pos), str(obj["chrNames"][ci]))
    if len(di) != len(pos):
        raise ValueError(f"DI y posiciones no coinciden en {obj['chrNames'][ci]}")
    order = np.argsort(pos)
    return pos[order], dm[:, order], di[order]


# -----------------------------------------------------------------------------
# File discovery and dataset metadata
# -----------------------------------------------------------------------------

@dataclass
class DatasetFiles:
    dataset_id: str
    thresholded: Path
    polarized: Path
    popmap: Path


def thresholded_prefix(path: Path) -> str:
    m = re.match(r"(.+?)\.thresholded(?:\(\d+\))?\.diemtype$", path.name, flags=re.I)
    if not m:
        raise ValueError(f"Nombre thresholded no reconocido: {path.name}")
    return m.group(1)


def find_case_insensitive(directory: Path, regex: str) -> List[Path]:
    patt = re.compile(regex, flags=re.I)
    return sorted([p for p in directory.iterdir() if p.is_file() and patt.fullmatch(p.name)])


def discover_datasets(input_dir: Path, thresholded_glob: str, logger: logging.Logger) -> List[DatasetFiles]:
    thresholded_files = sorted(input_dir.glob(thresholded_glob))
    if not thresholded_files:
        raise FileNotFoundError(f"No se encontraron archivos con {thresholded_glob} en {input_dir}")

    datasets: List[DatasetFiles] = []
    for thr in thresholded_files:
        prefix = thresholded_prefix(thr)
        pol_candidates = find_case_insensitive(
            input_dir,
            re.escape(prefix) + r"\.polarized(?:\(\d+\))?\.diemtype",
        )
        if not pol_candidates:
            raise FileNotFoundError(f"No se encontró polarized para {thr.name}")

        pop_patterns = [
            re.escape("popmap_" + prefix) + r"\.subset(?:\(\d+\))?\.txt",
            re.escape("PopMap_" + prefix) + r"\.subset(?:\(\d+\))?\.txt",
            re.escape("popmap_" + prefix) + r"(?:\(\d+\))?\.txt",
        ]
        pop_candidates: List[Path] = []
        for patt in pop_patterns:
            pop_candidates.extend(find_case_insensitive(input_dir, patt))
        # Duplicate-safe unique paths.
        pop_candidates = list(dict.fromkeys(pop_candidates))
        if not pop_candidates:
            # Last-resort prefix matching.
            pop_candidates = sorted(input_dir.glob(f"*{prefix}*popmap*.txt")) + sorted(input_dir.glob(f"popmap*{prefix}*.txt"))
            pop_candidates = list(dict.fromkeys(pop_candidates))
        if not pop_candidates:
            raise FileNotFoundError(f"No se encontró PopMap para {thr.name}; se esperaba popmap_{prefix}.subset.txt")

        datasets.append(DatasetFiles(prefix, thr.resolve(), pol_candidates[0].resolve(), pop_candidates[0].resolve()))
        logger.info("Dataset descubierto: %s | thresholded=%s | polarized=%s | popmap=%s",
                    prefix, thr.name, pol_candidates[0].name, pop_candidates[0].name)
    return datasets


def detect_ploidy(dataset_id: str, popmap: pd.DataFrame, obj: Dict[str, Any]) -> Tuple[str, str]:
    upper = dataset_id.upper()
    if upper.startswith("HAP") or "_HAP_" in upper:
        return "haploid", "filename"
    if upper.startswith("DIP") or "_DIP_" in upper:
        return "diploid", "filename"

    groups = " ".join(popmap["group"].astype(str).str.upper().tolist())
    if "DIP" in groups and "HAP" not in groups:
        return "diploid", "popmap"
    if "HAP" in groups and "DIP" not in groups:
        return "haploid", "popmap"

    tokens = []
    for ci in range(len(obj["DMBC"])):
        pos, dm, _ = canonical_contig(obj, ci)
        tokens.extend(np.unique(dm).tolist())
    token_set = set(int(x) for x in tokens if np.isfinite(x))
    if 2 in token_set:
        return "diploid", "state_token_2"
    return "haploid", "no_heterozygous_token"


def infer_threshold(obj: Dict[str, Any], dataset_id: str) -> float:
    if obj.get("threshold") is not None:
        try:
            return float(obj["threshold"])
        except Exception:
            pass
    m = re.search(r"(-\d+(?:\.\d+)?)$", dataset_id)
    if m:
        return float(m.group(1))
    all_di = np.concatenate([np.asarray(x, dtype=float) for x in obj["DIByChr"]])
    return float(np.nanmin(all_di))


# -----------------------------------------------------------------------------
# State orientation and ancestry dosage
# -----------------------------------------------------------------------------

def concatenate_matrix(obj: Dict[str, Any]) -> np.ndarray:
    mats = []
    for ci in range(len(obj["DMBC"])):
        _, dm, _ = canonical_contig(obj, ci)
        mats.append(dm)
    return np.concatenate(mats, axis=1) if mats else np.empty((len(obj["indNames"]), 0))


def lineage_label(population: str) -> str:
    """Etiqueta biológica legible para referencias conocidas; conserva nombres desconocidos."""
    pop = normalize_pop(population)
    aliases = {
        "CON": "Centro",
        "PUC": "Sur",
        "QCZ": "Norte",
        "RIT": "Centro",
    }
    return aliases.get(pop, pop)


def parse_reference_rules(text: str) -> List[Tuple[str, str, str]]:
    """Parsea reglas obligatorias: 'patron=REF_A,REF_B;otro*=REF_A,REF_B'."""
    rows: List[Tuple[str, str, str]] = []
    for item in str(text or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or "," not in item:
            raise ValueError(
                "REFERENCE_RULES inválido. Use patron=REF_A,REF_B;patron2=REF_A,REF_B"
            )
        pattern, pair = item.split("=", 1)
        a, b = pair.split(",", 1)
        pattern = pattern.strip()
        ref_a, ref_b = normalize_pop(a), normalize_pop(b)
        if not pattern:
            raise ValueError("REFERENCE_RULES contiene un patrón vacío")
        if ref_a == ref_b:
            raise ValueError(f"Regla {pattern}: las dos referencias deben ser distintas")
        rows.append((pattern, ref_a, ref_b))
    if not rows:
        raise ValueError(
            "REFERENCE_RULES está vacío. Las referencias deben declararse explícitamente en el runner."
        )
    return rows


def select_fixed_reference_populations(
    dataset_id: str,
    observed_pops: Sequence[str],
    rules_text: str,
) -> Tuple[str, str, str]:
    """Selecciona referencias solo desde reglas explícitas; nunca las infiere."""
    observed = [normalize_pop(x) for x in observed_pops]
    observed_set = set(observed)
    matches = [
        (pattern, ref_a, ref_b)
        for pattern, ref_a, ref_b in parse_reference_rules(rules_text)
        if fnmatchcase(dataset_id.upper(), pattern.upper())
    ]
    if len(matches) == 0:
        raise ValueError(
            f"No existe una regla de referencias para {dataset_id}. Observadas={observed}. "
            "Agregue una entrada explícita en REFERENCE_RULES del runner."
        )
    if len(matches) > 1:
        patterns = [m[0] for m in matches]
        raise ValueError(
            f"Más de una regla de referencias coincide con {dataset_id}: {patterns}. "
            "Use patrones mutuamente excluyentes."
        )
    pattern, ref_a, ref_b = matches[0]
    missing = [ref for ref in (ref_a, ref_b) if ref not in observed_set]
    if missing:
        raise ValueError(
            f"Regla {pattern}={ref_a},{ref_b} no coincide con las poblaciones observadas {observed}; "
            f"faltan {missing}."
        )
    return ref_a, ref_b, f"fixed_runner_rule:{pattern}"


def infer_reference_tokens(
    obj: Dict[str, Any],
    pop_by_ind: np.ndarray,
    reference_a: str,
    reference_b: str,
) -> Tuple[int, int, Optional[int], pd.DataFrame]:
    matrix = concatenate_matrix(obj)
    nonmissing_tokens = sorted(int(x) for x in np.unique(matrix) if int(x) != 0)
    homo_tokens = [x for x in nonmissing_tokens if x != 2]
    if len(homo_tokens) < 2:
        homo_tokens = [min(nonmissing_tokens), max(nonmissing_tokens)]
    homo_tokens = sorted(set(homo_tokens))
    if len(homo_tokens) != 2:
        raise ValueError(f"No fue posible identificar dos polos homocigotos. Tokens={nonmissing_tokens}")

    rows = []
    ref_major: Dict[str, int] = {}
    for role, ref in [("A", reference_a), ("B", reference_b)]:
        vals = matrix[pop_by_ind == ref]
        called = vals[vals != 0]
        if called.size == 0:
            raise ValueError(f"Referencia {ref} sin estados llamados")
        fractions = {tok: float(np.mean(called == tok)) for tok in homo_tokens}
        major = int(max(fractions, key=fractions.get))
        ref_major[ref] = major
        rows.append({
            "reference_role": role,
            "reference_population": ref,
            "reference_lineage": lineage_label(ref),
            "n_called_states": int(called.size),
            **{f"fraction_token_{tok}": fractions[tok] for tok in homo_tokens},
            "major_homozygous_token": major,
            "fraction_heterozygous_token_2": float(np.mean(called == 2)) if 2 in nonmissing_tokens else 0.0,
        })

    token_a = int(ref_major[reference_a])
    token_b = int(ref_major[reference_b])
    if token_a == token_b:
        raise ValueError(f"{reference_a} y {reference_b} fueron asignadas al mismo polo")
    hetero_token = 2 if 2 in nonmissing_tokens and 2 not in {token_a, token_b} else None
    stats = pd.DataFrame(rows)
    stats["reference_A_population"] = reference_a
    stats["reference_B_population"] = reference_b
    stats["reference_A_token"] = token_a
    stats["reference_B_token"] = token_b
    stats["heterozygous_token"] = hetero_token
    return token_a, token_b, hetero_token, stats


def reference_a_dosage(
    values: np.ndarray,
    token_a: int,
    token_b: int,
    hetero_token: Optional[int],
) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    out[values == token_a] = 1.0
    out[values == token_b] = 0.0
    if hetero_token is not None:
        out[values == hetero_token] = 0.5
    return out


def donor_dosage(values: np.ndarray, donor_token: int, recipient_token: int, hetero_token: Optional[int]) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    out[values == donor_token] = 1.0
    out[values == recipient_token] = 0.0
    if hetero_token is not None:
        out[values == hetero_token] = 0.5
    return out


def mixed_pattern_match(population: str, patterns_text: str) -> bool:
    patterns = [x.strip() for x in str(patterns_text or "").split(",") if x.strip()]
    return any(fnmatchcase(population.upper(), p.upper()) for p in patterns)


def assign_focal_backgrounds(
    dataset_id: str,
    obj: Dict[str, Any],
    pop_by_ind: np.ndarray,
    focal_pops: Sequence[str],
    reference_a: str,
    reference_b: str,
    token_a: int,
    token_b: int,
    hetero_token: Optional[int],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Asigna background por población o por individuo cuando el focal es bimodal."""
    matrix = concatenate_matrix(obj)
    names = list(obj["indNames"])
    dosage_by_ind = []
    for i in range(len(names)):
        vals = matrix[i]
        called = vals[vals != 0]
        d = reference_a_dosage(called, token_a, token_b, hetero_token)
        dosage_by_ind.append(float(np.nanmean(d)) if d.size else np.nan)

    assignment_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    for pop in focal_pops:
        idx = np.where(pop_by_ind == pop)[0]
        scores = np.asarray([dosage_by_ind[i] for i in idx], dtype=float)
        forced_split = mixed_pattern_match(pop, args.mixed_focal_patterns)
        bimodal = bool(
            np.any(scores >= args.individual_baseline_high)
            and np.any(scores <= args.individual_baseline_low)
        )
        split = bool(args.auto_split_mixed_focals and (forced_split or bimodal))
        population_mean = float(np.nanmean(scores)) if len(scores) else np.nan

        local_assignments = []
        for i, score in zip(idx, scores):
            sample = str(names[i])
            if split:
                if score >= args.individual_baseline_high:
                    recipient_ref, donor_ref = reference_a, reference_b
                    analysis_group = f"{pop}_{reference_a}-like"
                    status = "included_reference_A_like"
                    include = True
                elif score <= args.individual_baseline_low:
                    recipient_ref, donor_ref = reference_b, reference_a
                    analysis_group = f"{pop}_{reference_b}-like"
                    status = "included_reference_B_like"
                    include = True
                else:
                    recipient_ref = donor_ref = ""
                    analysis_group = f"{pop}_intermediate"
                    status = "excluded_ambiguous_background"
                    include = False
                method = "individual_bimodal_split"
            else:
                recipient_ref = reference_a if population_mean >= 0.5 else reference_b
                donor_ref = reference_b if recipient_ref == reference_a else reference_a
                analysis_group = pop
                status = "included_population_baseline"
                include = True
                method = "population_mean"

            row = {
                "dataset": dataset_id,
                "sample": sample,
                "source_population": pop,
                "analysis_group": analysis_group,
                "reference_A_population": reference_a,
                "reference_B_population": reference_b,
                "mean_reference_A_ancestry_dosage": score,
                "mean_reference_B_ancestry_dosage": 1.0 - score if np.isfinite(score) else np.nan,
                "recipient_reference_population": recipient_ref,
                "donor_reference_population": donor_ref,
                "recipient_lineage": lineage_label(recipient_ref) if recipient_ref else "",
                "donor_lineage": lineage_label(donor_ref) if donor_ref else "",
                "introgression_direction_compatible": (
                    f"{lineage_label(donor_ref)}→{lineage_label(recipient_ref)}" if recipient_ref else ""
                ),
                "baseline_assignment_method": method,
                "source_population_split": split,
                "assignment_status": status,
                "include_in_block_call": include,
            }
            assignment_rows.append(row)
            local_assignments.append(row)

        included = pd.DataFrame([r for r in local_assignments if r["include_in_block_call"]])
        if not included.empty:
            for group_name, g in included.groupby("analysis_group", sort=False):
                first = g.iloc[0]
                mean_a = float(g.mean_reference_A_ancestry_dosage.mean())
                group_rows.append({
                    "dataset": dataset_id,
                    "population": group_name,
                    "source_population": pop,
                    "n_individuals": int(len(g)),
                    "reference_A_population": reference_a,
                    "reference_B_population": reference_b,
                    "mean_reference_A_ancestry_dosage": mean_a,
                    "mean_reference_B_ancestry_dosage": 1.0 - mean_a,
                    "recipient_reference_population": first.recipient_reference_population,
                    "donor_reference_population": first.donor_reference_population,
                    "recipient_lineage": first.recipient_lineage,
                    "donor_lineage": first.donor_lineage,
                    "introgression_direction_compatible": first.introgression_direction_compatible,
                    "baseline_assignment_method": first.baseline_assignment_method,
                    "source_population_split": bool(first.source_population_split),
                    "baseline_margin": abs(mean_a - (1.0 - mean_a)),
                })

    return pd.DataFrame(assignment_rows), pd.DataFrame(group_rows)


# -----------------------------------------------------------------------------
# Block geometry and QC
# -----------------------------------------------------------------------------

def reciprocal_overlap(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    if inter <= 0:
        return 0.0
    return min(inter / (a1 - a0 + 1), inter / (b1 - b0 + 1))


def shared_fraction(a: Set[int], b: Set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def position_set(text: Any) -> Set[int]:
    if pd.isna(text) or str(text).strip() == "":
        return set()
    return {int(x) for x in str(text).split(",") if str(x).strip()}


def complete_link_compatible(candidate: pd.Series, members: List[pd.Series], ro_thr: float, snp_thr: float) -> bool:
    cpos = position_set(candidate["support_positions"])
    for member in members:
        ro = reciprocal_overlap(int(candidate.start), int(candidate.end), int(member.start), int(member.end))
        sf = shared_fraction(cpos, position_set(member["support_positions"]))
        if ro < ro_thr and sf < snp_thr:
            return False
    return True


def make_parameter_tag(args: argparse.Namespace) -> str:
    return (
        f"DIauto_gap{args.max_gap_bp}bp_min{args.min_support_snps}SNP_"
        f"min{args.min_length_bp}bp_support{fraction_tag(args.min_support_fraction)}_"
        f"miss{fraction_tag(args.max_missing_fraction)}_ref{fraction_tag(args.min_reference_coherence)}_"
        f"RO{fraction_tag(args.consensus_min_reciprocal_overlap)}_shared{fraction_tag(args.consensus_min_shared_fraction)}_"
        f"split{fraction_tag(args.individual_baseline_low)}to{fraction_tag(args.individual_baseline_high)}_fixedrefs"
    )


def dataset_parameter_tag(dataset_id: str, threshold: float, args: argparse.Namespace) -> str:
    return (
        f"{safe_name(dataset_id)}__DI{num_tag(threshold)}__gap{args.max_gap_bp}bp__"
        f"min{args.min_support_snps}SNP__min{args.min_length_bp}bp__"
        f"support{fraction_tag(args.min_support_fraction)}__miss{fraction_tag(args.max_missing_fraction)}__"
        f"ref{fraction_tag(args.min_reference_coherence)}__"
        f"split{fraction_tag(args.individual_baseline_low)}to{fraction_tag(args.individual_baseline_high)}__fixedrefs"
    )


def run_dataset(ds: DatasetFiles, args: argparse.Namespace, outdir: Path, logger: logging.Logger) -> Dict[str, Any]:
    obj = canonicalize_diemtype(ds.thresholded)
    polarized = canonicalize_diemtype(ds.polarized)
    pm = parse_popmap(ds.popmap)

    sample_to_pop = dict(zip(pm["sample"], pm["population"]))
    inds = np.asarray(obj["indNames"], dtype=object)
    missing = [str(x) for x in inds if str(x) not in sample_to_pop]
    if missing:
        raise ValueError(f"Muestras del DiemType ausentes del PopMap: {missing[:20]}")
    pop_by_ind = np.asarray([sample_to_pop[str(x)] for x in inds], dtype=object)
    observed_pops = list(dict.fromkeys(pop_by_ind.tolist()))

    reference_a, reference_b, reference_source = select_fixed_reference_populations(
        ds.dataset_id, observed_pops, args.reference_rules
    )
    focal_source_pops = [p for p in observed_pops if p not in {reference_a, reference_b}]
    if not focal_source_pops:
        raise ValueError(
            f"No existen poblaciones focales distintas de las referencias {reference_a}/{reference_b}"
        )

    ploidy, ploidy_source = detect_ploidy(ds.dataset_id, pm, obj)
    threshold = infer_threshold(obj, ds.dataset_id)
    dataset_tag = dataset_parameter_tag(ds.dataset_id, threshold, args)
    contrast = f"{reference_a}-{reference_b}"

    n_sites = int(sum(len(np.asarray(x)) for x in obj["posByChr"]))
    n_pre = int(sum(len(np.asarray(x)) for x in polarized["posByChr"]))
    all_di = np.concatenate([np.asarray(x, dtype=float) for x in obj["DIByChr"]])
    all_tokens = []
    for ci in range(len(obj["DMBC"])):
        _, dm, _ = canonical_contig(obj, ci)
        all_tokens.extend(np.unique(dm).tolist())
    tokens = sorted(int(x) for x in set(all_tokens))

    audit_rows = [
        {"dataset": ds.dataset_id, "check": "samples_match_popmap", "status": "PASS" if set(obj["indNames"]) == set(pm["sample"]) else "FAIL", "value": f"DiemType={len(obj['indNames'])}; PopMap={len(pm)}"},
        {"dataset": ds.dataset_id, "check": "no_smoothing_recorded", "status": "PASS" if obj.get("smoothScale") is None else "WARN", "value": str(obj.get("smoothScale"))},
        {"dataset": ds.dataset_id, "check": "contig_matrix_absent_or_ignored", "status": "PASS", "value": str(obj.get("contigMatrix") is not None)},
        {"dataset": ds.dataset_id, "check": "all_DI_at_or_above_threshold", "status": "PASS" if np.all(all_di >= threshold - 1e-9) else "FAIL", "value": f"minDI={float(np.nanmin(all_di))}; threshold={threshold}"},
        {"dataset": ds.dataset_id, "check": "ploidy_detected", "status": "PASS", "value": f"{ploidy} via {ploidy_source}"},
        {"dataset": ds.dataset_id, "check": "reference_populations", "status": "PASS", "value": f"A={reference_a}; B={reference_b}; source={reference_source}"},
        {"dataset": ds.dataset_id, "check": "state_tokens", "status": "PASS", "value": str(tokens)},
        {"dataset": ds.dataset_id, "check": "individual_background_thresholds", "status": "PASS", "value": f"low={args.individual_baseline_low}; high={args.individual_baseline_high}; mixed_patterns={args.mixed_focal_patterns}"},
    ]
    audit = pd.DataFrame(audit_rows)
    if (audit["status"] == "FAIL").any():
        raise RuntimeError(f"Auditoría crítica fallida: {audit[audit.status == 'FAIL'].to_dict('records')}")

    token_a, token_b, hetero_token, ref_stats = infer_reference_tokens(
        obj, pop_by_ind, reference_a, reference_b
    )
    ref_stats.insert(0, "dataset", ds.dataset_id)
    ref_stats.insert(1, "ploidy", ploidy)

    assignments, baselines = assign_focal_backgrounds(
        ds.dataset_id, obj, pop_by_ind, focal_source_pops,
        reference_a, reference_b, token_a, token_b, hetero_token, args
    )
    if baselines.empty:
        raise ValueError("Ningún individuo focal recibió un background parental inequívoco")
    assignment_map = assignments.set_index("sample").to_dict("index")
    baseline_map = baselines.set_index("population").to_dict("index")

    summary = pd.DataFrame([{
        "dataset": ds.dataset_id,
        "contrast": contrast,
        "reference_A_population": reference_a,
        "reference_B_population": reference_b,
        "reference_inference_source": reference_source,
        "ploidy": ploidy,
        "ploidy_detection_source": ploidy_source,
        "n_individuals": len(inds),
        "n_contigs": len(obj["chrNames"]),
        "n_sites_polarized": n_pre,
        "n_sites_thresholded": n_sites,
        "retained_fraction": n_sites / n_pre if n_pre else np.nan,
        "DI_threshold": threshold,
        "smoothScale_recorded": obj.get("smoothScale"),
        "state_tokens": ",".join(map(str, tokens)),
        "focal_populations": ",".join(focal_source_pops),
        "analysis_groups": ",".join(baselines.population.astype(str)),
        "n_focal_included": int(assignments.include_in_block_call.sum()),
        "n_focal_ambiguous_excluded": int((~assignments.include_in_block_call).sum()),
        "thresholded_file": str(ds.thresholded),
        "polarized_file": str(ds.polarized),
        "popmap_file": str(ds.popmap),
        "thresholded_sha256": sha256_file(ds.thresholded),
        "polarized_sha256": sha256_file(ds.polarized),
        "popmap_sha256": sha256_file(ds.popmap),
    }])

    genome = concatenate_matrix(obj)
    individual_rows = []
    for i, sample in enumerate(inds):
        vals = genome[i]
        called = vals[vals != 0]
        a = reference_a_dosage(called, token_a, token_b, hetero_token)
        assignment = assignment_map.get(str(sample), {})
        individual_rows.append({
            "dataset": ds.dataset_id,
            "contrast": contrast,
            "ploidy": ploidy,
            "sample": str(sample),
            "population": str(pop_by_ind[i]),
            "analysis_group": assignment.get("analysis_group", "REFERENCE"),
            "assignment_status": assignment.get("assignment_status", "reference_population"),
            "recipient_reference_population": assignment.get("recipient_reference_population", ""),
            "donor_reference_population": assignment.get("donor_reference_population", ""),
            "HI_diempy": float(obj["HIs"][i]) if i < len(obj["HIs"]) else np.nan,
            "n_sites_thresholded": int(vals.size),
            "n_called": int(called.size),
            "missing_fraction": float(np.mean(vals == 0)),
            "mean_reference_A_ancestry_dosage": float(np.nanmean(a)) if a.size else np.nan,
            "mean_reference_B_ancestry_dosage": float(1.0 - np.nanmean(a)) if a.size else np.nan,
            "heterozygous_state_fraction_called": float(np.mean(called == hetero_token)) if hetero_token is not None and called.size else 0.0,
        })
    individual_summary = pd.DataFrame(individual_rows)

    contig_rows: List[Dict[str, Any]] = []
    snp_rows: List[Dict[str, Any]] = []
    block_rows: List[Dict[str, Any]] = []
    block_counter = 0

    ref_a_idx = np.where(pop_by_ind == reference_a)[0]
    ref_b_idx = np.where(pop_by_ind == reference_b)[0]

    for ci, chrom in enumerate(obj["chrNames"]):
        pos, dm, di = canonical_contig(obj, ci)
        clen = int(obj["chrLengths"][ci])
        gaps = np.diff(pos)
        contig_rows.append({
            "dataset": ds.dataset_id,
            "contrast": contrast,
            "ploidy": ploidy,
            "contig": chrom,
            "contig_length_bp": clen,
            "n_thresholded_sites": len(pos),
            "first_thresholded_site": int(pos.min()) if len(pos) else np.nan,
            "last_thresholded_site": int(pos.max()) if len(pos) else np.nan,
            "median_interSNP_gap_bp": float(np.median(gaps)) if len(gaps) else np.nan,
            "p95_interSNP_gap_bp": float(np.quantile(gaps, 0.95)) if len(gaps) else np.nan,
            "max_interSNP_gap_bp": int(gaps.max()) if len(gaps) else np.nan,
        })

        ref_a_d = reference_a_dosage(dm[ref_a_idx], token_a, token_b, hetero_token)
        ref_b_d = reference_a_dosage(dm[ref_b_idx], token_a, token_b, hetero_token)

        for i, sample in enumerate(inds):
            sample_name = str(sample)
            if sample_name not in assignment_map:
                continue
            assignment = assignment_map[sample_name]
            if not assignment["include_in_block_call"]:
                continue

            source_pop = str(pop_by_ind[i])
            analysis_group = str(assignment["analysis_group"])
            recipient_ref_pop = str(assignment["recipient_reference_population"])
            donor_ref_pop = str(assignment["donor_reference_population"])
            recipient_lineage = str(assignment["recipient_lineage"])
            donor_lineage = str(assignment["donor_lineage"])
            recipient_token = token_a if recipient_ref_pop == reference_a else token_b
            donor_token = token_a if donor_ref_pop == reference_a else token_b
            direction = (
                f"{donor_lineage}-like ancestry in {analysis_group} "
                f"({assignment['introgression_direction_compatible']} compatible)"
            )

            evidence_specs: List[Tuple[str, np.ndarray]] = [
                ("haploid_donor_haplotype" if ploidy == "haploid" else "diploid_two_donor_copies", np.where(dm[i] == donor_token)[0]),
            ]
            if ploidy == "diploid" and hetero_token is not None:
                evidence_specs.append(("diploid_one_donor_copy_heterozygous", np.where(dm[i] == hetero_token)[0]))

            for evidence_layer, support_idx in evidence_specs:
                for si in support_idx:
                    snp_rows.append({
                        "dataset": ds.dataset_id,
                        "contrast": contrast,
                        "ploidy": ploidy,
                        "evidence_layer": evidence_layer,
                        "sample": sample_name,
                        "source_population": source_pop,
                        "population": analysis_group,
                        "recipient_reference_population": recipient_ref_pop,
                        "donor_reference_population": donor_ref_pop,
                        "recipient_lineage": recipient_lineage,
                        "donor_lineage": donor_lineage,
                        "direction": direction,
                        "contig": chrom,
                        "position": int(pos[si]),
                        "DI": float(di[si]),
                        "state_token": int(dm[i, si]),
                    })

                if len(support_idx) == 0:
                    continue
                support_pos = pos[support_idx]
                cluster_ids = np.cumsum(np.r_[True, np.diff(support_pos) > args.max_gap_bp])
                for gid in np.unique(cluster_ids):
                    idx = support_idx[cluster_ids == gid]
                    sp = pos[idx]
                    start, end = int(sp.min()), int(sp.max())
                    length_bp = end - start + 1
                    if len(idx) < args.min_support_snps or length_bp < args.min_length_bp:
                        continue

                    interval_idx = np.where((pos >= start) & (pos <= end))[0]
                    vals = dm[i, interval_idx]
                    called = vals[vals != 0]
                    support_token = donor_token if evidence_layer != "diploid_one_donor_copy_heterozygous" else hetero_token
                    support_fraction = float(np.mean(called == support_token)) if called.size else np.nan
                    d_dosage = donor_dosage(called, donor_token, recipient_token, hetero_token)
                    donor_dosage_mean = float(np.nanmean(d_dosage)) if d_dosage.size else np.nan
                    missing_fraction = float(np.mean(vals == 0)) if vals.size else np.nan

                    ref_a_interval = dm[np.ix_(ref_a_idx, interval_idx)]
                    ref_b_interval = dm[np.ix_(ref_b_idx, interval_idx)]
                    a_in_a = reference_a_dosage(ref_a_interval[ref_a_interval != 0], token_a, token_b, hetero_token)
                    a_in_b = reference_a_dosage(ref_b_interval[ref_b_interval != 0], token_a, token_b, hetero_token)
                    ref_a_coherence = float(np.nanmean(a_in_a)) if a_in_a.size else np.nan
                    ref_b_coherence = float(1.0 - np.nanmean(a_in_b)) if a_in_b.size else np.nan
                    recipient_ref_coherence = ref_a_coherence if recipient_ref_pop == reference_a else ref_b_coherence
                    donor_ref_coherence = ref_a_coherence if donor_ref_pop == reference_a else ref_b_coherence

                    block_counter += 1
                    block_rows.append({
                        "individual_block_id": f"{safe_name(ds.dataset_id)}_IB_{block_counter:06d}",
                        "dataset": ds.dataset_id,
                        "contrast": contrast,
                        "reference_A_population": reference_a,
                        "reference_B_population": reference_b,
                        "ploidy": ploidy,
                        "evidence_layer": evidence_layer,
                        "sample": sample_name,
                        "source_population": source_pop,
                        "population": analysis_group,
                        "recipient_reference_population": recipient_ref_pop,
                        "donor_reference_population": donor_ref_pop,
                        "recipient_lineage": recipient_lineage,
                        "donor_lineage": donor_lineage,
                        "direction": direction,
                        "contig": chrom,
                        "start": start,
                        "end": end,
                        "length_bp": length_bp,
                        "contig_length_bp": clen,
                        "contig_fraction": length_bp / clen if clen else np.nan,
                        "contig_percent": 100.0 * length_bp / clen if clen else np.nan,
                        "n_support_snps": int(len(idx)),
                        "n_thresholded_sites_interval": int(len(interval_idx)),
                        "n_called_sites_interval": int(called.size),
                        "n_missing_sites_interval": int(np.sum(vals == 0)),
                        "support_fraction_called": support_fraction,
                        "donor_ancestry_dosage_mean_called": donor_dosage_mean,
                        "missing_fraction": missing_fraction,
                        "max_internal_support_gap_bp": int(np.diff(sp).max()) if len(sp) > 1 else 0,
                        "mean_internal_support_gap_bp": float(np.diff(sp).mean()) if len(sp) > 1 else 0.0,
                        "DI_support_mean": float(np.mean(di[idx])),
                        "DI_support_median": float(np.median(di[idx])),
                        "DI_support_min": float(np.min(di[idx])),
                        "DI_support_max": float(np.max(di[idx])),
                        "reference_A_coherence": ref_a_coherence,
                        "reference_B_coherence": ref_b_coherence,
                        "recipient_reference_coherence": recipient_ref_coherence,
                        "donor_reference_coherence": donor_ref_coherence,
                        "support_positions": ",".join(map(str, sp.tolist())),
                    })

    raw = pd.DataFrame(block_rows)
    if raw.empty:
        qc = raw.copy()
    else:
        qc = raw.copy()
        qc["pass_support_fraction"] = qc["support_fraction_called"] >= args.min_support_fraction
        qc["pass_missingness"] = qc["missing_fraction"] <= args.max_missing_fraction
        qc["pass_reference_coherence"] = (
            (qc["recipient_reference_coherence"] >= args.min_reference_coherence)
            & (qc["donor_reference_coherence"] >= args.min_reference_coherence)
        )
        qc["pass_primary_QC"] = qc[["pass_support_fraction", "pass_missingness", "pass_reference_coherence"]].all(axis=1)

        def exclusion_reason(r: pd.Series) -> str:
            reasons = []
            if not r.pass_support_fraction:
                reasons.append(f"support_fraction<{args.min_support_fraction:g}")
            if not r.pass_missingness:
                reasons.append(f"missing_fraction>{args.max_missing_fraction:g}")
            if not r.pass_reference_coherence:
                reasons.append(f"reference_coherence<{args.min_reference_coherence:g}")
            return ";".join(reasons)

        qc["exclusion_reason"] = qc.apply(exclusion_reason, axis=1)

    logger.info(
        "%s | references=%s/%s | focal_groups=%s | ploidy=%s | threshold=%s | raw_blocks=%d | QC_pass=%d",
        ds.dataset_id, reference_a, reference_b, ",".join(baselines.population.astype(str)),
        ploidy, threshold, len(raw), int(qc.pass_primary_QC.sum()) if len(qc) else 0,
    )

    return {
        "dataset_id": ds.dataset_id,
        "dataset_tag": dataset_tag,
        "threshold": threshold,
        "ploidy": ploidy,
        "contrast": contrast,
        "focal_populations": focal_source_pops,
        "audit": audit,
        "dataset_summary": summary,
        "reference_polarization": ref_stats,
        "focal_baselines": baselines,
        "individual_assignments": assignments,
        "individual_summary": individual_summary,
        "contig_summary": pd.DataFrame(contig_rows),
        "discordant_snps": pd.DataFrame(snp_rows),
        "individual_blocks_raw": raw,
        "individual_blocks_qc": qc,
    }


# -----------------------------------------------------------------------------
# Consensus and cross-ploidy
# -----------------------------------------------------------------------------

def build_consensus(qc: pd.DataFrame, baselines: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if qc.empty or "pass_primary_QC" not in qc:
        return pd.DataFrame()
    passed = qc[qc["pass_primary_QC"]].copy()
    if passed.empty:
        return pd.DataFrame()

    n_by_pop = {(r.dataset, r.population): int(r.n_individuals) for _, r in baselines.iterrows()}
    rows = []
    counter = 0
    group_cols = [
        "dataset", "contrast", "ploidy", "evidence_layer", "source_population", "population",
        "recipient_reference_population", "donor_reference_population",
        "recipient_lineage", "donor_lineage", "direction", "contig",
    ]
    for keys, group in passed.groupby(group_cols, sort=False):
        clusters: List[List[pd.Series]] = []
        for _, candidate in group.sort_values(["start", "end", "sample"]).iterrows():
            placed = False
            for members in clusters:
                if complete_link_compatible(
                    candidate, members,
                    args.consensus_min_reciprocal_overlap,
                    args.consensus_min_shared_fraction,
                ):
                    members.append(candidate)
                    placed = True
                    break
            if not placed:
                clusters.append([candidate])

        for members in clusters:
            cdf = pd.DataFrame(members)
            counter += 1
            carriers = sorted(cdf["sample"].unique())
            analysis_group = keys[5]
            n_total = n_by_pop[(keys[0], analysis_group)]
            union_start, union_end = int(cdf.start.min()), int(cdf.end.max())
            core_start, core_end = int(cdf.start.max()), int(cdf.end.min())
            core_valid = core_start <= core_end
            recurrent = (
                len(carriers) >= args.recurrent_min_carriers
                and len(carriers) / n_total >= args.recurrent_min_carrier_fraction
            )
            if recurrent and core_valid:
                preferred_start, preferred_end, boundary = core_start, core_end, "shared_core"
            else:
                preferred_start, preferred_end, boundary = union_start, union_end, "union"
            preferred_length = preferred_end - preferred_start + 1
            contig_length = int(cdf.contig_length_bp.iloc[0])
            support_sets = [position_set(x) for x in cdf.support_positions]
            support_union = set().union(*support_sets) if support_sets else set()
            support_shared = set.intersection(*support_sets) if support_sets else set()

            rows.append({
                "candidate_block_id": f"PB_{counter:05d}",
                "dataset": keys[0],
                "contrast": keys[1],
                "ploidy": keys[2],
                "evidence_layer": keys[3],
                "source_population": keys[4],
                "population": analysis_group,
                "recipient_reference_population": keys[6],
                "donor_reference_population": keys[7],
                "recipient_lineage": keys[8],
                "donor_lineage": keys[9],
                "direction": keys[10],
                "contig": keys[11],
                "preferred_start": preferred_start,
                "preferred_end": preferred_end,
                "preferred_length_bp": preferred_length,
                "preferred_boundary_type": boundary,
                "union_start": union_start,
                "union_end": union_end,
                "union_length_bp": union_end - union_start + 1,
                "core_start": core_start if core_valid else np.nan,
                "core_end": core_end if core_valid else np.nan,
                "core_length_bp": core_end - core_start + 1 if core_valid else 0,
                "contig_length_bp": contig_length,
                "preferred_contig_fraction": preferred_length / contig_length if contig_length else np.nan,
                "preferred_contig_percent": 100.0 * preferred_length / contig_length if contig_length else np.nan,
                "n_focal_individuals": n_total,
                "n_carriers": len(carriers),
                "carrier_fraction": len(carriers) / n_total,
                "carrier_ids": ",".join(carriers),
                "recurrent": recurrent,
                "n_individual_segments": len(cdf),
                "mean_support_snps": float(cdf.n_support_snps.mean()),
                "min_support_snps": int(cdf.n_support_snps.min()),
                "max_support_snps": int(cdf.n_support_snps.max()),
                "support_snps_union": len(support_union),
                "support_snps_shared_all_carriers": len(support_shared),
                "mean_support_fraction_called": float(cdf.support_fraction_called.mean()),
                "min_support_fraction_called": float(cdf.support_fraction_called.min()),
                "mean_donor_ancestry_dosage": float(cdf.donor_ancestry_dosage_mean_called.mean()),
                "mean_missing_fraction": float(cdf.missing_fraction.mean()),
                "mean_recipient_reference_coherence": float(cdf.recipient_reference_coherence.mean()),
                "mean_donor_reference_coherence": float(cdf.donor_reference_coherence.mean()),
                "mean_DI_support": float(cdf.DI_support_mean.mean()),
                "minimum_DI_support": float(cdf.DI_support_min.min()),
                "member_individual_block_ids": ",".join(cdf.individual_block_id.astype(str)),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    def classify(r: pd.Series) -> str:
        if r.evidence_layer == "diploid_one_donor_copy_heterozygous":
            return "H_diploid_one_copy_support"
        ref_min = min(r.mean_recipient_reference_coherence, r.mean_donor_reference_coherence)
        if (
            r.recurrent
            and r.mean_support_fraction_called >= args.priority_A_min_support_fraction
            and ref_min >= args.priority_A_min_reference_coherence
        ):
            return "A_recurrent_high_coherence"
        if (
            not r.recurrent
            and r.mean_support_fraction_called >= args.priority_B_single_min_support_fraction
            and r.mean_support_snps >= args.priority_B_single_min_support_snps
            and ref_min >= args.priority_A_min_reference_coherence
        ):
            return "B_single_carrier_strong"
        return "C_candidate_requires_validation"

    out["priority_class"] = out.apply(classify, axis=1)
    out["interpretation"] = np.where(
        out.evidence_layer == "diploid_one_donor_copy_heterozygous",
        "diploid one-donor-copy ancestry support; secondary evidence",
        "ancestry-discordant candidate; introgression-compatible pending independent validation",
    )
    return out.sort_values(["priority_class", "dataset", "population", "contig", "preferred_start"]).reset_index(drop=True)


def best_overlap(query: pd.Series, candidates: pd.DataFrame) -> Optional[Tuple[pd.Series, float, bool]]:
    best = None
    for _, cand in candidates.iterrows():
        ro = reciprocal_overlap(
            int(query.preferred_start), int(query.preferred_end),
            int(cand.preferred_start), int(cand.preferred_end),
        )
        core_overlap = False
        if pd.notna(query.core_start) and pd.notna(cand.core_start):
            core_overlap = max(int(query.core_start), int(cand.core_start)) <= min(int(query.core_end), int(cand.core_end))
        if ro <= 0 and not core_overlap:
            continue
        score = ro + (1.0 if core_overlap else 0.0)
        if best is None or score > best[0]:
            best = (score, cand, ro, core_overlap)
    if best is None:
        return None
    return best[1], best[2], best[3]


def cross_ploidy_replication(consensus: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if consensus.empty:
        return pd.DataFrame()
    hap = consensus[(consensus.ploidy == "haploid") & (consensus.evidence_layer == "haploid_donor_haplotype")]
    dip_primary = consensus[(consensus.ploidy == "diploid") & (consensus.evidence_layer == "diploid_two_donor_copies")]
    dip_het = consensus[(consensus.ploidy == "diploid") & (consensus.evidence_layer == "diploid_one_donor_copy_heterozygous")]
    rows = []

    for _, h in hap.iterrows():
        same_primary = dip_primary[
            (dip_primary.population == h.population)
            & (dip_primary.donor_lineage == h.donor_lineage)
            & (dip_primary.contig == h.contig)
            & (dip_primary.contrast == h.contrast)
        ]
        same_het = dip_het[
            (dip_het.population == h.population)
            & (dip_het.donor_lineage == h.donor_lineage)
            & (dip_het.contig == h.contig)
            & (dip_het.contrast == h.contrast)
        ]
        bp = best_overlap(h, same_primary)
        bh = best_overlap(h, same_het)
        pro, pcore = (bp[1], bp[2]) if bp else (0.0, False)
        hro, hcore = (bh[1], bh[2]) if bh else (0.0, False)
        if bp and (pro >= args.cross_ploidy_min_overlap or pcore):
            cls = "strong_replication_diploid_two_copy"
        elif bh and (hro >= args.cross_ploidy_min_overlap or hcore):
            cls = "replication_diploid_one_copy"
        elif bp or bh:
            cls = "weak_coordinate_overlap"
        else:
            cls = "haploid_only"
        rows.append({
            "contrast": h.contrast,
            "population": h.population,
            "donor_lineage": h.donor_lineage,
            "contig": h.contig,
            "haploid_block_id": h.candidate_block_id,
            "diploid_primary_block_id": bp[0].candidate_block_id if bp else "",
            "diploid_heterozygous_block_id": bh[0].candidate_block_id if bh else "",
            "reciprocal_overlap_primary": pro,
            "core_overlap_primary": pcore,
            "reciprocal_overlap_heterozygous": hro,
            "core_overlap_heterozygous": hcore,
            "replication_class": cls,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def export_excel(path: Path, tables: Mapping[str, pd.DataFrame], logger: logging.Logger) -> None:
    try:
        import openpyxl  # noqa: F401
    except Exception:
        logger.warning("openpyxl no disponible; se omite Excel consolidado")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet = safe_name(name)[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = openpyxl.styles.Font(bold=True, color="FFFFFF")
                cell.fill = openpyxl.styles.PatternFill("solid", fgColor="1F4E78")
            for col in ws.columns:
                letter = col[0].column_letter
                max_len = min(max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2, 45)
                ws.column_dimensions[letter].width = max(10, max_len)


def print_terminal_summary(
    run_dir: Path,
    dataset_summaries: pd.DataFrame,
    raw: pd.DataFrame,
    qc: pd.DataFrame,
    consensus: pd.DataFrame,
    cross: pd.DataFrame,
) -> None:
    line = "=" * 110
    print("\n" + line)
    print("RESUMEN FINAL: BLOQUES ANCESTRY-DISCORDANT DESDE DIEMPY, SIN SMOOTHING NI HMM")
    print(line)
    cols = ["dataset", "contrast", "reference_A_population", "reference_B_population", "ploidy", "DI_threshold", "n_individuals", "n_sites_thresholded", "analysis_groups"]
    print("\nDatasets procesados:")
    print(dataset_summaries[cols].to_string(index=False))

    if raw.empty:
        print("\nNo se detectaron segmentos geométricos.")
    else:
        print("\nFlujo de bloques por dataset y capa:")
        raw_counts = raw.groupby(["dataset", "ploidy", "evidence_layer"]).size().rename("raw_blocks").reset_index()
        pass_counts = (
            qc[qc.pass_primary_QC]
            .groupby(["dataset", "ploidy", "evidence_layer"])
            .size().rename("QC_pass").reset_index()
        ) if not qc.empty else pd.DataFrame()
        flow = raw_counts.merge(pass_counts, on=["dataset", "ploidy", "evidence_layer"], how="left").fillna({"QC_pass": 0})
        flow["QC_pass"] = flow["QC_pass"].astype(int)
        print(flow.to_string(index=False))

    if not consensus.empty:
        print("\nRegiones poblacionales por clase:")
        counts = consensus.groupby(["ploidy", "priority_class"]).size().rename("n").reset_index()
        print(counts.to_string(index=False))
        top = consensus[consensus.priority_class.isin(["A_recurrent_high_coherence", "B_single_carrier_strong"])]
        if not top.empty:
            show_cols = [
                "candidate_block_id", "ploidy", "evidence_layer", "population", "contig",
                "preferred_start", "preferred_end", "preferred_length_bp", "contig_length_bp",
                "preferred_contig_percent", "direction", "n_carriers", "carrier_fraction", "priority_class",
            ]
            print("\nCandidatos principales:")
            print(top[show_cols].to_string(index=False, max_rows=100))
    if not cross.empty:
        print("\nReplicación haploide-diploide:")
        print(cross.replication_class.value_counts().rename_axis("class").reset_index(name="n").to_string(index=False))

    print("\nInterpretación: las regiones son ancestry-discordant y compatibles con introgresión; ChromoPainter NO se usó en esta llamada.")
    print(f"\nDIRECTORIO DE SALIDA: {run_dir}")
    print(line + "\n")


def dataframe_text_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_Sin datos._"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def build_report(
    path: Path,
    args: argparse.Namespace,
    summaries: pd.DataFrame,
    raw: pd.DataFrame,
    qc: pd.DataFrame,
    consensus: pd.DataFrame,
    cross: pd.DataFrame,
) -> None:
    lines = [
        "# Llamada de bloques ancestry-discordant desde diemPy",
        "",
        f"Generado: {utc_now()}",
        f"Versión: `{VERSION}`",
        "",
        "## Diseño",
        "",
        "- Objetos thresholded descubiertos automáticamente.",
        "- Ploidía y polos de referencia detectados automáticamente.",
        "- Poblaciones focales bimodales pueden dividirse por background individual.",
        "- Sin kernel smoothing y sin HMM.",
        "- ChromoPainter no se utilizó para seleccionar los bloques.",
        f"- Gap máximo: {args.max_gap_bp} bp.",
        f"- SNPs mínimos: {args.min_support_snps}.",
        f"- Longitud mínima: {args.min_length_bp} bp.",
        f"- Fracción mínima de soporte: {args.min_support_fraction}.",
        f"- Missingness máximo: {args.max_missing_fraction}.",
        f"- Coherencia mínima de referencias: {args.min_reference_coherence}.",
        f"- Cortes de background individual: ≤{args.individual_baseline_low} / ≥{args.individual_baseline_high}.",
        "",
        "## Datasets",
        "",
        dataframe_text_table(summaries),
        "",
        "## Flujo",
        "",
        f"- Segmentos geométricos: **{len(raw)}**.",
        f"- Segmentos que pasan QC: **{int(qc.pass_primary_QC.sum()) if not qc.empty else 0}**.",
        f"- Regiones poblacionales consolidadas: **{len(consensus)}**.",
    ]
    if not consensus.empty:
        lines += ["", "## Clasificación", "", dataframe_text_table(consensus.priority_class.value_counts().rename_axis("priority_class").reset_index(name="n"))]
    if not cross.empty:
        lines += ["", "## Replicación entre ploidías", "", dataframe_text_table(cross.replication_class.value_counts().rename_axis("replication_class").reset_index(name="n"))]
    lines += [
        "",
        "## Interpretación",
        "",
        "Las coordenadas corresponden al primer y último SNP de soporte y no son breakpoints exactos. Los bloques primarios y los tramos diploides heterocigotos se mantienen separados. Las regiones son compatibles con introgresión, pero requieren evaluación independiente mediante ChromoPainter, FST/dXY, genealogías locales y QC de mapeo.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bloques diemPy sin smoothing/HMM, con referencias biológicas fijas definidas en el runner")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--thresholded-glob", default="*.thresholded.diemtype")

    p.add_argument("--max-gap-bp", type=int, default=5000)
    p.add_argument("--min-support-snps", type=int, default=10)
    p.add_argument("--min-length-bp", type=int, default=10000)
    p.add_argument("--min-support-fraction", type=float, default=0.70)
    p.add_argument("--max-missing-fraction", type=float, default=0.25)
    p.add_argument("--min-reference-coherence", type=float, default=0.70)

    p.add_argument("--consensus-min-reciprocal-overlap", type=float, default=0.50)
    p.add_argument("--consensus-min-shared-fraction", type=float, default=0.50)
    p.add_argument("--recurrent-min-carriers", type=int, default=2)
    p.add_argument("--recurrent-min-carrier-fraction", type=float, default=0.20)

    p.add_argument("--priority-A-min-support-fraction", type=float, default=0.80)
    p.add_argument("--priority-A-min-reference-coherence", type=float, default=0.80)
    p.add_argument("--priority-B-single-min-support-fraction", type=float, default=0.90)
    p.add_argument("--priority-B-single-min-support-snps", type=int, default=15)
    p.add_argument("--cross-ploidy-min-overlap", type=float, default=0.50)
    p.add_argument("--reference-rules", required=True, help="Reglas obligatorias: patron=REF_A,REF_B;otro*=REF_A,REF_B")
    p.add_argument("--mixed-focal-patterns", default="ZTCN", help="Poblaciones a dividir por background individual; patrones separados por coma")
    p.add_argument("--individual-baseline-low", type=float, default=0.20)
    p.add_argument("--individual-baseline-high", type=float, default=0.80)
    p.add_argument("--auto-split-mixed-focals", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-label", default="diempy_blocks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.individual_baseline_low < args.individual_baseline_high <= 1.0):
        raise ValueError("Se requiere 0 <= individual_baseline_low < individual_baseline_high <= 1")
    input_dir = args.input_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    parameter_tag = make_parameter_tag(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{safe_name(args.run_label)}__{parameter_tag}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    log_path = run_dir / f"analysis_log__{parameter_tag}.log"
    logger = logging.getLogger("diempy_blocks")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("START version=%s", VERSION)
    logger.info("Command=%s", " ".join(sys.argv))
    logger.info("Input directory=%s", input_dir)
    logger.info("Output directory=%s", run_dir)
    logger.info("Parameter tag=%s", parameter_tag)
    logger.info("Chromopainter integration=NOT_PERFORMED")

    datasets = discover_datasets(input_dir, args.thresholded_glob, logger)
    all_results: List[Dict[str, Any]] = []
    errors = []
    manifest_rows = []

    for ds in datasets:
        for kind, path in [("thresholded", ds.thresholded), ("polarized", ds.polarized), ("popmap", ds.popmap)]:
            manifest_rows.append({
                "dataset": ds.dataset_id,
                "kind": kind,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        try:
            logger.info("PROCESSING dataset=%s", ds.dataset_id)
            result = run_dataset(ds, args, run_dir, logger)
            all_results.append(result)
        except Exception as exc:
            logger.error("FAILED dataset=%s error=%s", ds.dataset_id, exc)
            logger.error(traceback.format_exc())
            errors.append({"dataset": ds.dataset_id, "error": str(exc), "traceback": traceback.format_exc()})

    if not all_results:
        pd.DataFrame(errors).to_csv(run_dir / f"errors__{parameter_tag}.tsv", sep="\t", index=False)
        raise RuntimeError("Ningún dataset pudo procesarse; revise el log")

    # Combined tables.
    table_keys = [
        "audit", "dataset_summary", "reference_polarization", "focal_baselines", "individual_assignments",
        "individual_summary", "contig_summary", "discordant_snps",
        "individual_blocks_raw", "individual_blocks_qc",
    ]
    combined: Dict[str, pd.DataFrame] = {}
    for key in table_keys:
        frames = [r[key] for r in all_results if isinstance(r[key], pd.DataFrame) and not r[key].empty]
        combined[key] = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    consensus = build_consensus(combined["individual_blocks_qc"], combined["focal_baselines"], args)
    cross = cross_ploidy_replication(consensus, args)

    # Per-dataset output with dataset-specific DI in every filename.
    for result in all_results:
        dtag = result["dataset_tag"]
        data_tables = {
            "audit": result["audit"],
            "dataset_summary": result["dataset_summary"],
            "reference_polarization": result["reference_polarization"],
            "focal_baselines": result["focal_baselines"],
            "individual_assignments": result["individual_assignments"],
            "individual_genomewide": result["individual_summary"],
            "contig_summary": result["contig_summary"],
            "discordant_SNPs": result["discordant_snps"],
            "individual_blocks_raw": result["individual_blocks_raw"],
            "individual_blocks_QC": result["individual_blocks_qc"],
        }
        dcons = consensus[consensus.dataset == result["dataset_id"]].copy() if not consensus.empty else pd.DataFrame()
        data_tables["population_consensus_blocks"] = dcons
        data_tables["priority_candidates"] = dcons[dcons.priority_class.isin(["A_recurrent_high_coherence", "B_single_carrier_strong"])].copy() if not dcons.empty else pd.DataFrame()
        data_tables["diploid_one_copy_support"] = dcons[dcons.evidence_layer == "diploid_one_donor_copy_heterozygous"].copy() if not dcons.empty else pd.DataFrame()
        data_tables["excluded_individual_blocks"] = result["individual_blocks_qc"][~result["individual_blocks_qc"].pass_primary_QC].copy() if not result["individual_blocks_qc"].empty else pd.DataFrame()
        for label, df in data_tables.items():
            suffix = ".tsv.gz" if label == "discordant_SNPs" else ".tsv"
            write_tsv(df, run_dir / f"{dtag}__{label}{suffix}")

        # BED with all candidate consensus intervals.
        if not dcons.empty:
            bed = dcons[["contig", "preferred_start", "preferred_end", "candidate_block_id", "priority_class", "population", "direction"]].copy()
            bed.insert(1, "bed_start_0based", (bed.pop("preferred_start") - 1).clip(lower=0).astype(int))
            bed.insert(2, "bed_end_0based_halfopen", bed.pop("preferred_end").astype(int))
            write_tsv(bed, run_dir / f"{dtag}__population_consensus_blocks.bed")

    # Combined output also carries the global parameter tag.
    combined_tables = {
        "input_manifest": pd.DataFrame(manifest_rows),
        "dataset_audit": combined["audit"],
        "dataset_summary": combined["dataset_summary"],
        "reference_polarization": combined["reference_polarization"],
        "focal_baselines": combined["focal_baselines"],
        "individual_assignments": combined["individual_assignments"],
        "individual_genomewide": combined["individual_summary"],
        "contig_summary": combined["contig_summary"],
        "individual_blocks_raw": combined["individual_blocks_raw"],
        "individual_blocks_QC": combined["individual_blocks_qc"],
        "population_consensus_blocks": consensus,
        "priority_candidates": consensus[consensus.priority_class.isin(["A_recurrent_high_coherence", "B_single_carrier_strong"])].copy() if not consensus.empty else pd.DataFrame(),
        "diploid_one_copy_support": consensus[consensus.evidence_layer == "diploid_one_donor_copy_heterozygous"].copy() if not consensus.empty else pd.DataFrame(),
        "cross_ploidy_replication": cross,
        "errors": pd.DataFrame(errors),
    }
    for label, df in combined_tables.items():
        write_tsv(df, run_dir / f"ALL_DATASETS__{parameter_tag}__{label}.tsv")

    # Consolidated Excel and report.
    excel_path = run_dir / f"ALL_DATASETS__{parameter_tag}__results.xlsx"
    export_excel(excel_path, combined_tables, logger)
    report_path = run_dir / f"RESULTS_REPORT__{parameter_tag}.md"
    build_report(report_path, args, combined["dataset_summary"], combined["individual_blocks_raw"], combined["individual_blocks_qc"], consensus, cross)

    # Copy exact code used. No separate parameters file is produced.
    shutil.copy2(Path(__file__), run_dir / f"SCRIPT_USED__{parameter_tag}.py")

    inventory = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            inventory.append({"file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_tsv(pd.DataFrame(inventory), run_dir / f"OUTPUT_INVENTORY__{parameter_tag}.tsv")

    logger.info(
        "FINISH datasets=%d raw=%d QC=%d consensus=%d cross_ploidy=%d errors=%d",
        len(all_results),
        len(combined["individual_blocks_raw"]),
        int(combined["individual_blocks_qc"].pass_primary_QC.sum()) if not combined["individual_blocks_qc"].empty else 0,
        len(consensus),
        len(cross),
        len(errors),
    )
    print_terminal_summary(run_dir, combined["dataset_summary"], combined["individual_blocks_raw"], combined["individual_blocks_qc"], consensus, cross)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
