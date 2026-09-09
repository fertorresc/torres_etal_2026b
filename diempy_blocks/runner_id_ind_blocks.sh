#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# EDITAR SOLO ESTE BLOQUE
# =============================================================================

INPUT_DIR="/media/server/a77f75fe-fd07-402e-84d7-a7341c29141c/fertorres/segundo_capitulo/integration_analyses/Julio"
OUTPUT_ROOT="${INPUT_DIR}/diempy_blocks_unsmoothed"
PYTHON_SCRIPT="${INPUT_DIR}/diempy_blocks_fixed_references.py"

THRESHOLDED_GLOB="*.thresholded.diemtype"
RUN_LABEL="Julio_diempy_unsmoothed_fixed_refs_v4"

# -----------------------------------------------------------------------------
# REFERENCIAS BIOLÓGICAS FIJAS Y OBLIGATORIAS
# -----------------------------------------------------------------------------
# Formato: patron=REFERENCIA_A,REFERENCIA_B;patron2=REFERENCIA_A,REFERENCIA_B
#
# Centro–Sur: siempre CON y PUC.
# Norte–Centro: siempre QCZ y RIT.
#
# El script NO infiere referencias desde los datos ni desde el nombre.
# El nombre solo se usa para escoger una de estas reglas explícitas.
# Si un dataset no coincide con exactamente una regla, el análisis se detiene.
REFERENCE_RULES="HAP_CON_LIL_LOB_PUC*=CON,PUC;DIP_CON_LIL_LOB_PUC*=CON,PUC;HAP_CON_LLI_CVG_PUC*=CON,PUC;DIP_CON_LLI_CVG_PUC*=CON,PUC;HAP_CON_to_PUC*=CON,PUC;DIP_CON_to_PUC*=CON,PUC;HAP_QCZ_to_RIT*=QCZ,RIT;DIP_QCZ_to_RIT*=QCZ,RIT"

# Criterios geométricos del bloque.
MAX_GAP_BP=5000
MIN_SUPPORT_SNPS=10
MIN_LENGTH_BP=10000

# QC interno.
MIN_SUPPORT_FRACTION=0.70
MAX_MISSING_FRACTION=0.25
MIN_REFERENCE_COHERENCE=0.70

# Consolidación de portadores.
CONSENSUS_MIN_RECIPROCAL_OVERLAP=0.50
CONSENSUS_MIN_SHARED_FRACTION=0.50
RECURRENT_MIN_CARRIERS=2
RECURRENT_MIN_CARRIER_FRACTION=0.20

# Clasificación de prioridad.
PRIORITY_A_MIN_SUPPORT_FRACTION=0.80
PRIORITY_A_MIN_REFERENCE_COHERENCE=0.80
PRIORITY_B_SINGLE_MIN_SUPPORT_FRACTION=0.90
PRIORITY_B_SINGLE_MIN_SUPPORT_SNPS=15

# Replicación entre haploides y diploides.
CROSS_PLOIDY_MIN_OVERLAP=0.50

# Solo ZTCN se divide por background individual QCZ-like/RIT-like.
# LIL, LOB, LLI y CVG mantienen un único background poblacional CON-like o PUC-like.
MIXED_FOCAL_PATTERNS="ZTCN"
AUTO_SPLIT_MIXED_FOCALS=1
INDIVIDUAL_BASELINE_LOW=0.20
INDIVIDUAL_BASELINE_HIGH=0.80

# =============================================================================
# NO EDITAR DESDE AQUÍ
# =============================================================================

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: no existe INPUT_DIR: $INPUT_DIR" >&2
  exit 1
fi
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "ERROR: no existe PYTHON_SCRIPT: $PYTHON_SCRIPT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

cat <<EOF2
================================================================================
EJECUCIÓN DIEMPY SIN SMOOTHING NI HMM — REFERENCIAS FIJAS
================================================================================
INPUT_DIR                         = $INPUT_DIR
OUTPUT_ROOT                       = $OUTPUT_ROOT
THRESHOLDED_GLOB                  = $THRESHOLDED_GLOB
REFERENCE_RULES                   = $REFERENCE_RULES
MAX_GAP_BP                        = $MAX_GAP_BP
MIN_SUPPORT_SNPS                  = $MIN_SUPPORT_SNPS
MIN_LENGTH_BP                     = $MIN_LENGTH_BP
MIN_SUPPORT_FRACTION              = $MIN_SUPPORT_FRACTION
MAX_MISSING_FRACTION              = $MAX_MISSING_FRACTION
MIN_REFERENCE_COHERENCE           = $MIN_REFERENCE_COHERENCE
CONSENSUS_MIN_RECIPROCAL_OVERLAP  = $CONSENSUS_MIN_RECIPROCAL_OVERLAP
CONSENSUS_MIN_SHARED_FRACTION     = $CONSENSUS_MIN_SHARED_FRACTION
RECURRENT_MIN_CARRIERS            = $RECURRENT_MIN_CARRIERS
RECURRENT_MIN_CARRIER_FRACTION    = $RECURRENT_MIN_CARRIER_FRACTION
MIXED_FOCAL_PATTERNS              = $MIXED_FOCAL_PATTERNS
INDIVIDUAL_BASELINE_LOW/HIGH      = $INDIVIDUAL_BASELINE_LOW / $INDIVIDUAL_BASELINE_HIGH
CHROMOPAINTER                     = NO SE USA EN ESTA LLAMADA
================================================================================
EOF2

SPLIT_FLAG="--auto-split-mixed-focals"
if [[ "$AUTO_SPLIT_MIXED_FOCALS" != "1" ]]; then
  SPLIT_FLAG="--no-auto-split-mixed-focals"
fi

python "$PYTHON_SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --thresholded-glob "$THRESHOLDED_GLOB" \
  --run-label "$RUN_LABEL" \
  --reference-rules "$REFERENCE_RULES" \
  --max-gap-bp "$MAX_GAP_BP" \
  --min-support-snps "$MIN_SUPPORT_SNPS" \
  --min-length-bp "$MIN_LENGTH_BP" \
  --min-support-fraction "$MIN_SUPPORT_FRACTION" \
  --max-missing-fraction "$MAX_MISSING_FRACTION" \
  --min-reference-coherence "$MIN_REFERENCE_COHERENCE" \
  --consensus-min-reciprocal-overlap "$CONSENSUS_MIN_RECIPROCAL_OVERLAP" \
  --consensus-min-shared-fraction "$CONSENSUS_MIN_SHARED_FRACTION" \
  --recurrent-min-carriers "$RECURRENT_MIN_CARRIERS" \
  --recurrent-min-carrier-fraction "$RECURRENT_MIN_CARRIER_FRACTION" \
  --priority-A-min-support-fraction "$PRIORITY_A_MIN_SUPPORT_FRACTION" \
  --priority-A-min-reference-coherence "$PRIORITY_A_MIN_REFERENCE_COHERENCE" \
  --priority-B-single-min-support-fraction "$PRIORITY_B_SINGLE_MIN_SUPPORT_FRACTION" \
  --priority-B-single-min-support-snps "$PRIORITY_B_SINGLE_MIN_SUPPORT_SNPS" \
  --cross-ploidy-min-overlap "$CROSS_PLOIDY_MIN_OVERLAP" \
  --mixed-focal-patterns "$MIXED_FOCAL_PATTERNS" \
  --individual-baseline-low "$INDIVIDUAL_BASELINE_LOW" \
  --individual-baseline-high "$INDIVIDUAL_BASELINE_HIGH" \
  "$SPLIT_FLAG"

LATEST_RUN=$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
if [[ -n "${LATEST_RUN:-}" && -d "$LATEST_RUN" ]]; then
  RUN_BASENAME=$(basename "$LATEST_RUN")
  cp -f "$0" "$LATEST_RUN/RUNNER_USED__${RUN_BASENAME}.sh"
  echo "Runner copiado en: $LATEST_RUN/RUNNER_USED__${RUN_BASENAME}.sh"
fi
