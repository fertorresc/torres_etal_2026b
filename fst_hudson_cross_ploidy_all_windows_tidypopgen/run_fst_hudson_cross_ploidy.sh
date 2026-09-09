#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE="/media/server/a77f75fe-fd07-402e-84d7-a7341c29141c1/fertorres/segundo_capitulo/integration_analyses/Julio"
export FST_BASE_DIR="${FST_BASE_DIR:-$DEFAULT_BASE}"
export FST_N_CORES="${FST_N_CORES:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/check_inputs_and_environment.sh"
cd "$FST_BASE_DIR"
Rscript "$SCRIPT_DIR/fst_hudson_cross_ploidy_all_windows_tidypopgen.R"
