#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE="/media/server/a77f75fe-fd07-402e-84d7-a7341c29141c1/fertorres/segundo_capitulo/integration_analyses/Julio"
BASE="${FST_BASE_DIR:-$DEFAULT_BASE}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VCF="$BASE/vcf_maf_md_100kpb_contigs_pseudodip.vcf.gz"

fail=0
check_file() {
  local f="$1"
  if [[ -s "$f" ]]; then
    printf '[OK] %s\n' "$f"
  else
    printf '[ERROR] No existe o está vacío: %s\n' "$f" >&2
    fail=1
  fi
}

command -v Rscript >/dev/null 2>&1 || { echo '[ERROR] Rscript no está disponible en PATH.' >&2; fail=1; }
check_file "$VCF"
check_file "$SCRIPT_DIR/inputs/PopMap_HAP.txt"
check_file "$SCRIPT_DIR/inputs/PopMap_DIP.txt"
check_file "$SCRIPT_DIR/inputs/diemPy_priority_blocks_for_FST.tsv"
check_file "$SCRIPT_DIR/inputs/ZTCN_reassignment_HAP_from_diemPy.tsv"
check_file "$SCRIPT_DIR/inputs/ZTCN_reassignment_DIP_from_diemPy.tsv"

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

Rscript - <<'RS'
required <- c("tidypopgen", "dplyr", "tibble")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Paquetes R faltantes: ", paste(missing, collapse = ", "))
v <- as.character(packageVersion("tidypopgen"))
if (v != "0.4.4") stop("Se requiere tidypopgen 0.4.4; versión encontrada: ", v)
cat("[OK] tidypopgen 0.4.4, dplyr y tibble disponibles.\n")
RS

echo '[OK] Verificación previa completada.'
