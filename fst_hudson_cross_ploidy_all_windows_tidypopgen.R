#!/usr/bin/env Rscript

# ==============================================================================
# Hudson FST cruzado: todas las regiones diemPy evaluadas en HAP y DIP
#
# Diseño:
#   Cada intervalo candidato de diemPy se evalúa en ambos conjuntos de ploidía.
#   Además, los intervalos solapados de HAP y DIP se armonizan como regiones
#   genómicas comunes para producir una comparación fila-a-fila.
#
# Para cada intervalo/región:
#   FST del intervalo focal versus FST del mismo contig sin el intervalo focal.
# Unidad poblacional: todos los individuos de cada población, nunca solo carriers.
# HAP y DIP se calculan por separado con exactamente las mismas coordenadas.
# Los valores negativos de FST se preservan.
# ==============================================================================

options(stringsAsFactors = FALSE, warn = 1)

# ----------------------------- CONFIGURACIÓN ---------------------------------
BASE_DIR <- Sys.getenv(
  "FST_BASE_DIR",
  unset = "/media/server/a77f75fe-fd07-402e-84d7-a7341c29141c1/fertorres/segundo_capitulo/integration_analyses/Julio"
)

VCF_HAP <- file.path(BASE_DIR, "vcf_maf_md_100kpb_contigs_pseudodip.vcf.gz")
VCF_DIP <- file.path(BASE_DIR, "vcf_maf_md_100kpb_contigs_pseudodip.vcf.gz")

# Los popmaps validados se definen después de resolver SCRIPT_DIR/INPUT_DIR.
POPMAP_HAP <- NA_character_
POPMAP_DIP <- NA_character_

N_CORES <- as.integer(Sys.getenv("FST_N_CORES", unset = "30"))
MIN_LOCI_REGION <- 5L
MIN_INDIVIDUALS_PER_POP <- 2L
REQUIRED_TIDYPOPGEN_VERSION <- "0.4.4"
STRICT_ZTCN_ASSIGNMENT <- TRUE

# Conteos validados manualmente el 2026-08-03.
EXPECTED_ZTCN_COUNTS <- list(
  HAP = c(QCZ = 16L, RIT = 14L, EXCLUDED = 0L),
  DIP = c(QCZ = 2L, RIT = 20L, EXCLUDED = 9L)
)

# NULL = usar todos los bloques presentes en priority_candidates.
# Ejemplo para restringir:
# PRIORITY_CLASSES <- c("A_recurrent_high_coherence")
PRIORITY_CLASSES <- NULL

# -------------------------- RUTAS DEL PAQUETE --------------------------------
get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0L) {
    return(normalizePath(".", mustWork = TRUE))
  }
  normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
}

SCRIPT_PATH <- get_script_path()
SCRIPT_DIR <- if (dir.exists(SCRIPT_PATH)) SCRIPT_PATH else dirname(SCRIPT_PATH)
INPUT_DIR <- file.path(SCRIPT_DIR, "inputs")

BLOCKS_FILE <- file.path(INPUT_DIR, "diemPy_priority_blocks_for_FST.tsv")
ZTCN_ASSIGN_HAP <- file.path(INPUT_DIR, "ZTCN_reassignment_HAP_from_diemPy.tsv")
ZTCN_ASSIGN_DIP <- file.path(INPUT_DIR, "ZTCN_reassignment_DIP_from_diemPy.tsv")
POPMAP_HAP <- file.path(INPUT_DIR, "PopMap_HAP.txt")
POPMAP_DIP <- file.path(INPUT_DIR, "PopMap_DIP.txt")

RUN_ID <- format(Sys.time(), "%Y%m%d_%H%M%S")
OUTROOT <- file.path(BASE_DIR, "FST_Hudson_cross_ploidy_all_windows_tidypopgen")
OUTDIR <- file.path(OUTROOT, paste0("run_", RUN_ID))
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
BACKING_DIR <- file.path(OUTDIR, "backing")
dir.create(BACKING_DIR, recursive = TRUE, showWarnings = FALSE)
LOG_FILE <- file.path(OUTDIR, "run.log")

# ------------------------------- UTILIDADES ----------------------------------
log_msg <- function(...) {
  txt <- paste0(..., collapse = "")
  line <- sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), txt)
  cat(line, "\n", file = LOG_FILE, append = TRUE)
  message(line)
}

abort_run <- function(...) {
  txt <- paste0(..., collapse = "")
  log_msg("ERROR: ", txt)
  stop(txt, call. = FALSE)
}

write_tsv <- function(x, path) {
  utils::write.table(
    x,
    file = path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    col.names = TRUE,
    na = "NA"
  )
}

assert_file <- function(path, label, candidate_pattern = NULL) {
  if (file.exists(path)) return(invisible(TRUE))
  msg <- paste0(label, " no encontrado: ", path)
  if (!is.null(candidate_pattern) && dir.exists(BASE_DIR)) {
    candidates <- list.files(
      BASE_DIR,
      pattern = candidate_pattern,
      full.names = TRUE,
      recursive = FALSE,
      ignore.case = TRUE
    )
    if (length(candidates) > 0L) {
      msg <- paste0(msg, "\nCandidatos encontrados:\n", paste(candidates, collapse = "\n"))
    }
  }
  abort_run(msg)
}

normalize_population <- function(x) {
  x <- toupper(trimws(as.character(x)))
  x <- gsub("[[:space:]]+", "", x)
  x <- sub("_(HAP|HAPLOID|DIP|DIPLOID)$", "", x)

  out <- x
  out[grepl("^CONS($|_)", x)] <- "CON"
  out[grepl("^CON($|_)", x)] <- "CON"
  out[grepl("^POS($|_)", x)] <- "POS"
  out[grepl("^PUC($|_)", x)] <- "PUC"
  out[grepl("^QCZ($|_)", x)] <- "QCZ"
  out[grepl("^RIT($|_)", x)] <- "RIT"
  out[grepl("^ZTCN($|_)", x)] <- "ZTCN"
  out[grepl("^LLI($|_)", x)] <- "LLI"
  out[grepl("^CVG($|_)", x)] <- "CVG"
  out[grepl("^LIL($|_)", x)] <- "LIL"
  out[grepl("^LOB($|_)", x)] <- "LOB"
  out
}

read_popmap <- function(path, ploidy_label) {
  x <- utils::read.table(
    path,
    header = FALSE,
    sep = "",
    quote = "",
    comment.char = "",
    fill = TRUE,
    stringsAsFactors = FALSE
  )

  if (ncol(x) < 2L) {
    abort_run("El popmap ", path, " debe tener al menos dos columnas: ID y población.")
  }

  x <- x[, 1:2, drop = FALSE]
  first_1 <- tolower(trimws(as.character(x[1, 1])))
  first_2 <- tolower(trimws(as.character(x[1, 2])))
  if (first_1 %in% c("id", "sample", "individual", "ind") ||
      first_2 %in% c("population", "pop", "poblacion")) {
    x <- x[-1, , drop = FALSE]
  }

  names(x) <- c("id", "population_raw")
  x$id <- trimws(as.character(x$id))
  x$population_raw <- trimws(as.character(x$population_raw))
  x <- x[nzchar(x$id) & nzchar(x$population_raw), , drop = FALSE]

  if (anyDuplicated(x$id)) {
    dup <- unique(x$id[duplicated(x$id)])
    abort_run("IDs duplicados en ", ploidy_label, " popmap: ", paste(dup, collapse = ", "))
  }

  x$ploidy_dataset <- ploidy_label
  x$population_before_reassignment <- normalize_population(x$population_raw)
  x
}

read_ztcn_assignments <- function(path, ploidy_label) {
  x <- utils::read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
  required <- c("sample", "analysis_group")
  missing_cols <- setdiff(required, names(x))
  if (length(missing_cols) > 0L) {
    abort_run(
      "Faltan columnas en asignaciones ZTCN ", ploidy_label, ": ",
      paste(missing_cols, collapse = ", ")
    )
  }

  if (!"final_population" %in% names(x)) {
    x$final_population <- NA_character_
    x$final_population[x$analysis_group == "ZTCN_QCZ-like"] <- "QCZ"
    x$final_population[x$analysis_group == "ZTCN_RIT-like"] <- "RIT"
  }

  if (!"include_in_fst" %in% names(x)) {
    x$include_in_fst <- !is.na(x$final_population) & nzchar(x$final_population)
  }

  x$sample <- as.character(x$sample)
  x$analysis_group <- as.character(x$analysis_group)
  x$final_population <- as.character(x$final_population)
  x$include_in_fst <- tolower(trimws(as.character(x$include_in_fst))) %in%
    c("true", "t", "1", "yes", "y")

  if (anyDuplicated(x$sample)) {
    dup <- unique(x$sample[duplicated(x$sample)])
    abort_run("Asignaciones ZTCN duplicadas en ", ploidy_label, ": ", paste(dup, collapse = ", "))
  }
  x
}

validate_ztcn_assignment_table <- function(popmap, assignments, ploidy_label) {
  ztcn_ids <- sort(popmap$id[popmap$population_before_reassignment == "ZTCN"])
  assignment_ids <- sort(assignments$sample)

  missing_assignments <- setdiff(ztcn_ids, assignment_ids)
  extra_assignments <- setdiff(assignment_ids, ztcn_ids)

  if (length(missing_assignments) > 0L || length(extra_assignments) > 0L) {
    abort_run(
      "La tabla de asignación ZTCN ", ploidy_label,
      " no coincide exactamente con los ZTCN del popmap.",
      if (length(missing_assignments) > 0L)
        paste0("\nSin asignación: ", paste(missing_assignments, collapse = ", "))
      else "",
      if (length(extra_assignments) > 0L)
        paste0("\nAsignaciones no presentes en popmap: ", paste(extra_assignments, collapse = ", "))
      else ""
    )
  }

  expected <- EXPECTED_ZTCN_COUNTS[[ploidy_label]]
  observed_qcz <- sum(assignments$include_in_fst & assignments$final_population == "QCZ", na.rm = TRUE)
  observed_rit <- sum(assignments$include_in_fst & assignments$final_population == "RIT", na.rm = TRUE)
  observed_excluded <- sum(!assignments$include_in_fst | is.na(assignments$final_population))

  observed <- c(QCZ = observed_qcz, RIT = observed_rit, EXCLUDED = observed_excluded)

  if (!identical(as.integer(observed), as.integer(expected))) {
    abort_run(
      "Conteos ZTCN inesperados en ", ploidy_label,
      ". Observados: ", paste(names(observed), observed, sep = "=", collapse = ", "),
      "; esperados: ", paste(names(expected), expected, sep = "=", collapse = ", ")
    )
  }

  log_msg(
    "Validación ZTCN ", ploidy_label, " superada: ",
    "QCZ-like=", observed[["QCZ"]], "; ",
    "RIT-like=", observed[["RIT"]], "; ",
    "excluidos=", observed[["EXCLUDED"]]
  )
  invisible(TRUE)
}

apply_ztcn_reassignment <- function(popmap, assignments, ploidy_label) {
  out <- popmap
  out$ztcn_analysis_group <- NA_character_
  out$final_population <- out$population_before_reassignment
  out$include_in_fst <- TRUE
  out$exclusion_reason <- NA_character_

  is_ztcn <- out$population_before_reassignment == "ZTCN"
  idx <- match(out$id[is_ztcn], assignments$sample)

  if (STRICT_ZTCN_ASSIGNMENT && anyNA(idx)) {
    missing_ids <- out$id[is_ztcn][is.na(idx)]
    abort_run(
      "Hay individuos ZTCN del popmap ", ploidy_label,
      " sin asignación QCZ-like/RIT-like/intermediate en diemPy: ",
      paste(missing_ids, collapse = ", ")
    )
  }

  valid <- is_ztcn
  valid[is_ztcn] <- !is.na(idx)
  assignment_rows <- idx[!is.na(idx)]
  target_rows <- which(is_ztcn)[!is.na(idx)]

  out$ztcn_analysis_group[target_rows] <- assignments$analysis_group[assignment_rows]
  out$final_population[target_rows] <- assignments$final_population[assignment_rows]
  out$include_in_fst[target_rows] <- assignments$include_in_fst[assignment_rows]

  excluded <- target_rows[!out$include_in_fst[target_rows] | is.na(out$final_population[target_rows])]
  if (length(excluded) > 0L) {
    out$include_in_fst[excluded] <- FALSE
    out$final_population[excluded] <- NA_character_
    out$exclusion_reason[excluded] <- "ZTCN_intermediate_or_ambiguous"
  }

  out$reassignment_action <- ifelse(
    out$population_before_reassignment != "ZTCN",
    "unchanged",
    ifelse(
      out$include_in_fst & out$final_population == "QCZ",
      "ZTCN_QCZ-like_to_QCZ",
      ifelse(
        out$include_in_fst & out$final_population == "RIT",
        "ZTCN_RIT-like_to_RIT",
        "ZTCN_excluded"
      )
    )
  )

  out
}

population_count_audit <- function(popmap_audit, ploidy_label) {
  before <- as.data.frame(table(popmap_audit$population_before_reassignment), stringsAsFactors = FALSE)
  names(before) <- c("population", "n_individuals")
  before$stage <- "before_reassignment"
  before$ploidy <- ploidy_label

  after_data <- popmap_audit[popmap_audit$include_in_fst & !is.na(popmap_audit$final_population), , drop = FALSE]
  after <- as.data.frame(table(after_data$final_population), stringsAsFactors = FALSE)
  names(after) <- c("population", "n_individuals")
  after$stage <- "after_reassignment"
  after$ploidy <- ploidy_label

  excluded <- data.frame(
    population = "EXCLUDED",
    n_individuals = sum(!popmap_audit$include_in_fst),
    stage = "after_reassignment",
    ploidy = ploidy_label,
    stringsAsFactors = FALSE
  )

  rbind(before, after, excluded)
}

prepare_gen_tibble <- function(vcf_path, popmap_audit, ploidy_label) {
  backing_prefix <- file.path(BACKING_DIR, paste0(tolower(ploidy_label), "_all_vcf"))
  log_msg(ploidy_label, ": importando VCF con gen_tibble(): ", vcf_path)

  gt_all <- tidypopgen::gen_tibble(
    vcf_path,
    parser = "cpp",
    n_cores = N_CORES,
    backingfile = backing_prefix,
    quiet = FALSE
  )

  vcf_ids <- as.character(gt_all$id)
  missing_in_vcf <- setdiff(popmap_audit$id, vcf_ids)
  if (length(missing_in_vcf) > 0L) {
    abort_run(
      ploidy_label, ": IDs del popmap ausentes en el VCF: ",
      paste(missing_in_vcf, collapse = ", ")
    )
  }

  extra_vcf <- setdiff(vcf_ids, popmap_audit$id)
  sample_audit <- data.frame(
    id = vcf_ids,
    in_ploidy_popmap = vcf_ids %in% popmap_audit$id,
    stringsAsFactors = FALSE
  )
  pm_idx_all <- match(sample_audit$id, popmap_audit$id)
  sample_audit$population_raw <- popmap_audit$population_raw[pm_idx_all]
  sample_audit$population_before_reassignment <- popmap_audit$population_before_reassignment[pm_idx_all]
  sample_audit$final_population <- popmap_audit$final_population[pm_idx_all]
  sample_audit$include_in_fst <- popmap_audit$include_in_fst[pm_idx_all]
  write_tsv(sample_audit, file.path(OUTDIR, paste0("sample_audit_all_vcf_samples_", ploidy_label, ".tsv")))

  if (length(extra_vcf) > 0L) {
    log_msg(ploidy_label, ": ", length(extra_vcf), " muestras del VCF no pertenecen a este popmap y serán ignoradas.")
  }

  gt <- dplyr::filter(gt_all, id %in% popmap_audit$id)
  pm_idx <- match(as.character(gt$id), popmap_audit$id)
  gt$population_raw <- popmap_audit$population_raw[pm_idx]
  gt$population_before_reassignment <- popmap_audit$population_before_reassignment[pm_idx]
  gt$population <- popmap_audit$final_population[pm_idx]
  gt$include_in_fst <- popmap_audit$include_in_fst[pm_idx]

  gt <- dplyr::filter(gt, include_in_fst & !is.na(population))

  if (ploidy_label == "HAP") {
    log_msg("HAP: detectando pseudohaploides en todos los loci.")
    gt <- tidypopgen::gt_pseudohaploid(gt, test_n_loci = NULL)
    ind_ploidy <- tidypopgen::indiv_ploidy(gt)
    if (!all(ind_ploidy == 1L)) {
      bad <- as.character(gt$id)[ind_ploidy != 1L]
      abort_run("HAP: individuos no detectados como pseudohaploides: ", paste(bad, collapse = ", "))
    }
  } else if (ploidy_label == "DIP") {
    ind_ploidy <- tidypopgen::indiv_ploidy(gt)
    if (!all(ind_ploidy == 2L)) {
      bad <- as.character(gt$id)[ind_ploidy != 2L]
      abort_run("DIP: individuos con ploidía distinta de 2: ", paste(bad, collapse = ", "))
    }
  } else {
    abort_run("Ploidía desconocida: ", ploidy_label)
  }

  ploidy_audit <- data.frame(
    id = as.character(gt$id),
    population = as.character(gt$population),
    inferred_ploidy = as.integer(tidypopgen::indiv_ploidy(gt)),
    stringsAsFactors = FALSE
  )
  write_tsv(ploidy_audit, file.path(OUTDIR, paste0("ploidy_audit_", ploidy_label, ".tsv")))

  tidypopgen::gt_save(
    gt,
    file_name = file.path(BACKING_DIR, paste0(tolower(ploidy_label), "_analysis_subset.gt")),
    quiet = FALSE
  )

  log_msg(
    ploidy_label, ": gen_tibble listo. Individuos = ", nrow(gt),
    "; loci = ", tidypopgen::count_loci(gt),
    "; show_ploidy = ", tidypopgen::show_ploidy(gt)
  )
  gt
}

read_blocks <- function(path) {
  x <- utils::read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
  required <- c(
    "candidate_block_id", "ploidy", "population", "contig",
    "preferred_start", "preferred_end", "priority_class"
  )
  missing_cols <- setdiff(required, names(x))
  if (length(missing_cols) > 0L) {
    abort_run("Faltan columnas en tabla de bloques: ", paste(missing_cols, collapse = ", "))
  }

  x$ploidy <- tolower(as.character(x$ploidy))
  x$preferred_start <- as.integer(x$preferred_start)
  x$preferred_end <- as.integer(x$preferred_end)
  x$contig <- as.character(x$contig)
  x$population <- as.character(x$population)

  if (!is.null(PRIORITY_CLASSES)) {
    x <- x[x$priority_class %in% PRIORITY_CLASSES, , drop = FALSE]
  }

  if (nrow(x) == 0L) abort_run("No quedaron bloques después del filtro PRIORITY_CLASSES.")
  if (any(x$preferred_end < x$preferred_start, na.rm = TRUE)) {
    abort_run("Hay bloques con preferred_end < preferred_start.")
  }
  x
}


# -------------------- REGIONES COMUNES ENTRE PLOIDÍAS -------------------------
comparison_from_population <- function(block_population) {
  pair <- local_pair_from_block(block_population)
  if (anyNA(pair)) return(NA_character_)
  paste(pair, collapse = "-")
}

natural_contig_number <- function(x) {
  out <- suppressWarnings(as.integer(sub(".*?([0-9]+)$", "\\1", as.character(x))))
  out[is.na(out)] <- .Machine$integer.max
  out
}

collapse_unique <- function(x, sep = ",") {
  x <- unique(as.character(x))
  x <- x[!is.na(x) & nzchar(x)]
  if (length(x) == 0L) return(NA_character_)
  paste(x, collapse = sep)
}

interval_string <- function(ids, starts, ends) {
  paste0(ids, "=", starts, "-", ends, collapse = ";")
}

add_comparison_columns <- function(blocks) {
  blocks$fst_comparison <- vapply(blocks$population, comparison_from_population, character(1))
  pair_list <- lapply(blocks$population, local_pair_from_block)
  blocks$fst_population_1 <- vapply(pair_list, `[[`, character(1), 1L)
  blocks$fst_population_2 <- vapply(pair_list, `[[`, character(1), 2L)
  if (anyNA(blocks$fst_comparison)) {
    bad <- blocks$candidate_block_id[is.na(blocks$fst_comparison)]
    abort_run("No se pudo asignar comparación local a: ", paste(bad, collapse = ", "))
  }
  blocks
}

intervals_overlap <- function(start1, end1, start2, end2) {
  start1 <= end2 & start2 <= end1
}

build_exact_windows <- function(blocks) {
  x <- blocks
  x$source_ploidy <- x$ploidy
  x$analysis_region_id <- as.character(x$candidate_block_id)
  x$analysis_start <- as.integer(x$preferred_start)
  x$analysis_end <- as.integer(x$preferred_end)
  x$analysis_length_bp <- x$analysis_end - x$analysis_start + 1L
  x$region_definition <- "exact_original_diemPy_interval"
  x$source_candidate_ids <- x$candidate_block_id
  x$n_source_blocks <- 1L

  hap_ids <- dip_ids <- character(nrow(x))
  detected_hap <- detected_dip <- logical(nrow(x))

  for (i in seq_len(nrow(x))) {
    same_context <- x$fst_comparison == x$fst_comparison[[i]] &
      x$contig == x$contig[[i]] &
      intervals_overlap(
        x$preferred_start, x$preferred_end,
        x$preferred_start[[i]], x$preferred_end[[i]]
      )
    h <- x[same_context & x$ploidy == "haploid", , drop = FALSE]
    d <- x[same_context & x$ploidy == "diploid", , drop = FALSE]
    detected_hap[[i]] <- nrow(h) > 0L
    detected_dip[[i]] <- nrow(d) > 0L
    hap_ids[[i]] <- collapse_unique(h$candidate_block_id)
    dip_ids[[i]] <- collapse_unique(d$candidate_block_id)
  }

  x$detected_by_diemPy_HAP <- detected_hap
  x$detected_by_diemPy_DIP <- detected_dip
  x$detection_class <- ifelse(
    detected_hap & detected_dip, "HAP_and_DIP",
    ifelse(detected_hap, "HAP_only", "DIP_only")
  )
  x$overlapping_candidate_ids_HAP <- hap_ids
  x$overlapping_candidate_ids_DIP <- dip_ids
  x$source_intervals_HAP <- ifelse(
    detected_hap,
    vapply(seq_len(nrow(x)), function(i) {
      same <- x$fst_comparison == x$fst_comparison[[i]] & x$contig == x$contig[[i]] &
        intervals_overlap(x$preferred_start, x$preferred_end, x$preferred_start[[i]], x$preferred_end[[i]]) &
        x$ploidy == "haploid"
      interval_string(x$candidate_block_id[same], x$preferred_start[same], x$preferred_end[same])
    }, character(1)),
    NA_character_
  )
  x$source_intervals_DIP <- ifelse(
    detected_dip,
    vapply(seq_len(nrow(x)), function(i) {
      same <- x$fst_comparison == x$fst_comparison[[i]] & x$contig == x$contig[[i]] &
        intervals_overlap(x$preferred_start, x$preferred_end, x$preferred_start[[i]], x$preferred_end[[i]]) &
        x$ploidy == "diploid"
      interval_string(x$candidate_block_id[same], x$preferred_start[same], x$preferred_end[same])
    }, character(1)),
    NA_character_
  )

  ord <- order(
    match(x$fst_comparison, c("QCZ-RIT", "LLI-CVG", "LIL-LOB")),
    natural_contig_number(x$contig), x$analysis_start, x$analysis_end,
    x$candidate_block_id
  )
  x <- x[ord, , drop = FALSE]
  rownames(x) <- NULL
  x$display_order <- seq_len(nrow(x))
  x
}

build_harmonized_regions <- function(blocks) {
  ord <- order(
    match(blocks$fst_comparison, c("QCZ-RIT", "LLI-CVG", "LIL-LOB")),
    natural_contig_number(blocks$contig), blocks$preferred_start,
    blocks$preferred_end, blocks$candidate_block_id
  )
  x <- blocks[ord, , drop = FALSE]
  rownames(x) <- NULL

  groups <- list()
  current <- integer(0)
  current_comp <- current_contig <- NA_character_
  current_end <- NA_integer_

  flush_group <- function(indices) {
    if (length(indices) > 0L) groups[[length(groups) + 1L]] <<- indices
  }

  for (i in seq_len(nrow(x))) {
    same_context <- length(current) > 0L &&
      x$fst_comparison[[i]] == current_comp &&
      x$contig[[i]] == current_contig
    overlaps_current_union <- same_context && x$preferred_start[[i]] <= current_end

    if (!overlaps_current_union) {
      flush_group(current)
      current <- i
      current_comp <- x$fst_comparison[[i]]
      current_contig <- x$contig[[i]]
      current_end <- x$preferred_end[[i]]
    } else {
      current <- c(current, i)
      current_end <- max(current_end, x$preferred_end[[i]], na.rm = TRUE)
    }
  }
  flush_group(current)

  out <- vector("list", length(groups))
  for (g in seq_along(groups)) {
    z <- x[groups[[g]], , drop = FALSE]
    hap <- z[z$ploidy == "haploid", , drop = FALSE]
    dip <- z[z$ploidy == "diploid", , drop = FALSE]
    detected_hap <- nrow(hap) > 0L
    detected_dip <- nrow(dip) > 0L

    out[[g]] <- data.frame(
      analysis_region_id = sprintf("HR_%05d", g),
      fst_comparison = z$fst_comparison[[1L]],
      fst_population_1 = z$fst_population_1[[1L]],
      fst_population_2 = z$fst_population_2[[1L]],
      contig = z$contig[[1L]],
      analysis_start = min(z$preferred_start, na.rm = TRUE),
      analysis_end = max(z$preferred_end, na.rm = TRUE),
      analysis_length_bp = max(z$preferred_end, na.rm = TRUE) - min(z$preferred_start, na.rm = TRUE) + 1L,
      region_definition = "union_of_overlapping_diemPy_intervals_same_contig_and_population_comparison",
      detected_by_diemPy_HAP = detected_hap,
      detected_by_diemPy_DIP = detected_dip,
      detection_class = ifelse(
        detected_hap & detected_dip, "HAP_and_DIP",
        ifelse(detected_hap, "HAP_only", "DIP_only")
      ),
      source_candidate_ids = collapse_unique(z$candidate_block_id),
      source_candidate_ids_HAP = collapse_unique(hap$candidate_block_id),
      source_candidate_ids_DIP = collapse_unique(dip$candidate_block_id),
      source_intervals_HAP = if (detected_hap) interval_string(hap$candidate_block_id, hap$preferred_start, hap$preferred_end) else NA_character_,
      source_intervals_DIP = if (detected_dip) interval_string(dip$candidate_block_id, dip$preferred_start, dip$preferred_end) else NA_character_,
      source_populations = collapse_unique(z$population),
      source_priority_classes = collapse_unique(z$priority_class),
      source_datasets = collapse_unique(z$dataset),
      n_source_blocks = nrow(z),
      n_source_blocks_HAP = nrow(hap),
      n_source_blocks_DIP = nrow(dip),
      contig_length_bp = suppressWarnings(max(z$contig_length_bp, na.rm = TRUE)),
      stringsAsFactors = FALSE
    )
  }

  result <- dplyr::bind_rows(out)
  result$display_order <- seq_len(nrow(result))
  result
}

local_pair_from_block <- function(block_population) {
  p <- as.character(block_population)
  if (p %in% c("QCZ", "RIT", "ZTCN_QCZ-like", "ZTCN_RIT-like")) return(c("QCZ", "RIT"))
  if (p %in% c("LLI", "CVG")) return(c("LLI", "CVG"))
  if (p %in% c("LIL", "LOB")) return(c("LIL", "LOB"))
  c(NA_character_, NA_character_)
}

calc_region_fst <- function(gt, loci_mask, pop1, pop2, region_label) {
  n_loci <- sum(loci_mask, na.rm = TRUE)
  base_result <- list(
    region = region_label,
    n_loci = as.integer(n_loci),
    n_individuals_pop1 = sum(as.character(gt$population) == pop1),
    n_individuals_pop2 = sum(as.character(gt$population) == pop2),
    fst = NA_real_,
    status = "NOT_CALCULATED",
    error = NA_character_
  )

  if (n_loci < MIN_LOCI_REGION) {
    base_result$status <- "INSUFFICIENT_LOCI"
    return(base_result)
  }

  if (base_result$n_individuals_pop1 < MIN_INDIVIDUALS_PER_POP ||
      base_result$n_individuals_pop2 < MIN_INDIVIDUALS_PER_POP) {
    base_result$status <- "INSUFFICIENT_INDIVIDUALS"
    return(base_result)
  }

  tryCatch({
    gt_region <- tidypopgen::select_loci_if(gt, loci_mask)
    gt_pair <- dplyr::filter(gt_region, population %in% c(pop1, pop2))
    gt_pair <- dplyr::group_by(gt_pair, population)

    fst_res <- tidypopgen::pairwise_pop_fst(
      gt_pair,
      type = "tidy",
      by_locus = FALSE,
      method = "Hudson",
      n_cores = N_CORES
    )

    if (nrow(fst_res) != 1L || !"value" %in% names(fst_res)) {
      base_result$status <- "UNEXPECTED_FST_OUTPUT"
      base_result$error <- paste(capture.output(str(fst_res)), collapse = " ")
      return(base_result)
    }

    base_result$fst <- as.numeric(fst_res$value[[1L]])
    base_result$status <- ifelse(is.finite(base_result$fst), "OK", "FST_NA_OR_NONFINITE")
    base_result
  }, error = function(e) {
    base_result$status <- "ERROR"
    base_result$error <- conditionMessage(e)
    base_result
  })
}


process_regions <- function(gt, regions, analysis_ploidy, analysis_set) {
  loci <- tidypopgen::show_loci(gt)
  if (!all(c("chromosome", "position") %in% names(loci))) {
    abort_run(analysis_ploidy, ": show_loci() no contiene chromosome y position.")
  }

  loci$chromosome <- as.character(loci$chromosome)
  loci$position <- as.integer(loci$position)

  out_list <- vector("list", nrow(regions))
  checkpoint <- file.path(
    OUTDIR,
    paste0("FST_Hudson_", analysis_set, "_", analysis_ploidy, "_PARTIAL.tsv")
  )

  for (i in seq_len(nrow(regions))) {
    b <- regions[i, , drop = FALSE]
    pop1 <- as.character(b$fst_population_1[[1L]])
    pop2 <- as.character(b$fst_population_2[[1L]])

    log_msg(
      analysis_set, " ", analysis_ploidy, " [", i, "/", nrow(regions), "] ",
      b$analysis_region_id[[1L]], " ", b$contig[[1L]], ":",
      b$analysis_start[[1L]], "-", b$analysis_end[[1L]],
      "; pair = ", pop1, " vs ", pop2
    )

    same_contig <- loci$chromosome == b$contig[[1L]]
    in_block <- same_contig &
      loci$position >= b$analysis_start[[1L]] &
      loci$position <= b$analysis_end[[1L]]
    in_background <- same_contig & !in_block

    block_res <- calc_region_fst(gt, in_block, pop1, pop2, "block")
    background_res <- calc_region_fst(gt, in_background, pop1, pop2, "contig_background")

    delta <- if (is.finite(block_res$fst) && is.finite(background_res$fst)) {
      block_res$fst - background_res$fst
    } else {
      NA_real_
    }

    relation <- if (is.na(delta)) {
      NA_character_
    } else if (delta < 0) {
      "FST_block_lower_than_background"
    } else if (delta > 0) {
      "FST_block_higher_than_background"
    } else {
      "FST_block_equal_to_background"
    }

    row <- as.data.frame(b, stringsAsFactors = FALSE)
    row$analysis_set <- analysis_set
    row$analysis_ploidy <- analysis_ploidy
    row$n_individuals_population_1 <- block_res$n_individuals_pop1
    row$n_individuals_population_2 <- block_res$n_individuals_pop2
    row$n_loci_block <- block_res$n_loci
    row$n_loci_contig_background <- background_res$n_loci
    row$fst_block_Hudson <- block_res$fst
    row$fst_contig_background_Hudson <- background_res$fst
    row$delta_fst_block_minus_background <- delta
    row$block_vs_background_relation <- relation
    row$block_status <- block_res$status
    row$background_status <- background_res$status
    row$block_error <- block_res$error
    row$background_error <- background_res$error
    row$background_definition <- "same_contig_excluding_only_focal_analysis_interval"
    row$individual_scope <- "all_individuals_in_each_population"
    row$negative_fst_handling <- "preserved"

    out_list[[i]] <- row
    write_tsv(dplyr::bind_rows(out_list[seq_len(i)]), checkpoint)
  }

  result <- dplyr::bind_rows(out_list)
  final_path <- file.path(
    OUTDIR,
    paste0("FST_Hudson_", analysis_set, "_", analysis_ploidy, ".tsv")
  )
  write_tsv(result, final_path)
  if (file.exists(checkpoint)) file.remove(checkpoint)
  result
}

make_wide_results <- function(regions, result_hap, result_dip) {
  out <- as.data.frame(regions, stringsAsFactors = FALSE)
  calc_cols <- c(
    "n_individuals_population_1", "n_individuals_population_2",
    "n_loci_block", "n_loci_contig_background",
    "fst_block_Hudson", "fst_contig_background_Hudson",
    "delta_fst_block_minus_background", "block_vs_background_relation",
    "block_status", "background_status", "block_error", "background_error"
  )

  append_ploidy <- function(base, result, suffix) {
    idx <- match(base$analysis_region_id, result$analysis_region_id)
    for (nm in calc_cols) {
      base[[paste0(nm, "_", suffix)]] <- result[[nm]][idx]
    }
    base
  }

  out <- append_ploidy(out, result_hap, "HAP")
  out <- append_ploidy(out, result_dip, "DIP")
  out$fst_block_HAP_minus_DIP <- out$fst_block_Hudson_HAP - out$fst_block_Hudson_DIP
  out$delta_HAP_minus_DIP <- out$delta_fst_block_minus_background_HAP -
    out$delta_fst_block_minus_background_DIP
  out$analysis_complete_both <-
    out$block_status_HAP == "OK" & out$background_status_HAP == "OK" &
    out$block_status_DIP == "OK" & out$background_status_DIP == "OK"
  out$phase_pattern <- ifelse(
    out$delta_fst_block_minus_background_HAP < 0 & out$delta_fst_block_minus_background_DIP < 0,
    "lower_FST_in_both",
    ifelse(
      out$delta_fst_block_minus_background_HAP < 0 & out$delta_fst_block_minus_background_DIP >= 0,
      "lower_FST_HAP_only",
      ifelse(
        out$delta_fst_block_minus_background_HAP >= 0 & out$delta_fst_block_minus_background_DIP < 0,
        "lower_FST_DIP_only",
        "not_lower_in_either"
      )
    )
  )
  out
}

write_aligned_dumbbell_plot <- function(wide, path) {
  keep_any <- is.finite(wide$fst_block_Hudson_HAP) |
    is.finite(wide$fst_contig_background_Hudson_HAP) |
    is.finite(wide$fst_block_Hudson_DIP) |
    is.finite(wide$fst_contig_background_Hudson_DIP)
  x <- wide[keep_any, , drop = FALSE]
  if (nrow(x) == 0L) return(invisible(NULL))

  all_values <- c(
    x$fst_block_Hudson_HAP, x$fst_contig_background_Hudson_HAP,
    x$fst_block_Hudson_DIP, x$fst_contig_background_Hudson_DIP
  )
  all_values <- all_values[is.finite(all_values)]
  xr <- range(c(0, 1, all_values), na.rm = TRUE)
  pad <- max(0.03, diff(xr) * 0.04)
  xlim <- c(xr[[1L]] - pad, xr[[2L]] + pad)

  n <- nrow(x)
  y <- rev(seq_len(n))
  detection_tag <- ifelse(
    x$detection_class == "HAP_and_DIP", "H+D",
    ifelse(x$detection_class == "HAP_only", "H", "D")
  )
  labels <- paste0(
    x$analysis_region_id, " [", x$fst_comparison, "; ", detection_tag, "] ",
    x$contig, ":", x$analysis_start, "-", x$analysis_end
  )

  height_px <- max(1800L, 80L * n + 500L)
  grDevices::png(path, width = 2700, height = height_px, res = 220)
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit({graphics::par(old_par); grDevices::dev.off()}, add = TRUE)
  graphics::par(mfrow = c(1, 2), oma = c(3, 2, 4, 1), xaxs = "i")

  comp_change <- which(x$fst_comparison[-1L] != x$fst_comparison[-n])
  separator_y <- if (length(comp_change) > 0L) y[comp_change] - 0.5 else numeric(0)

  panel <- function(block, background, title_text, show_labels) {
    graphics::par(mar = if (show_labels) c(4.5, 16.5, 3, 1.5) else c(4.5, 2.0, 3, 2.5))
    graphics::plot(
      NA, xlim = xlim, ylim = c(0.4, n + 0.6), yaxt = "n",
      xlab = expression(paste("Hudson ", F[ST])), ylab = "",
      main = title_text, bty = "l"
    )
    graphics::abline(v = 0, col = "grey85", lty = 3)
    if (length(separator_y) > 0L) graphics::abline(h = separator_y, col = "grey70", lty = 2)
    valid <- is.finite(block) & is.finite(background)
    graphics::segments(background[valid], y[valid], block[valid], y[valid], col = "#2C7FB8", lwd = 1.5)
    graphics::points(background[valid], y[valid], pch = 16, cex = 0.85, col = "#2C7FB8")
    graphics::points(block[valid], y[valid], pch = 16, cex = 0.85, col = "#F16913")
    if (show_labels) graphics::axis(2, at = y, labels = labels, las = 1, cex.axis = 0.58, tick = FALSE)
  }

  panel(x$fst_block_Hudson_HAP, x$fst_contig_background_Hudson_HAP, "HAP", TRUE)
  panel(x$fst_block_Hudson_DIP, x$fst_contig_background_Hudson_DIP, "DIP", FALSE)

  graphics::mtext(
    expression(paste("Hudson ", F[ST], " de las mismas regiones diemPy en HAP y DIP")),
    side = 3, outer = TRUE, line = 1.2, cex = 1.2, font = 2
  )
  graphics::mtext(
    "H = detectada por diemPy en haploides; D = detectada en diploides; H+D = detectada en ambos",
    side = 1, outer = TRUE, line = 1.0, cex = 0.75
  )
  graphics::legend(
    "bottom", inset = -0.17, xpd = NA, horiz = TRUE, bty = "n",
    legend = c("Resto del mismo contig", "Región diemPy"),
    pch = 16, col = c("#2C7FB8", "#F16913"), cex = 0.85
  )
}

# --------------------------------- INICIO -------------------------------------
cat("START\n", file = LOG_FILE)
log_msg("BASE_DIR: ", BASE_DIR)
log_msg("SCRIPT_DIR: ", SCRIPT_DIR)
log_msg("OUTDIR: ", OUTDIR)

assert_file(VCF_HAP, "VCF haploide", "\\.vcf(\\.gz)?$")
assert_file(VCF_DIP, "VCF diploide", "\\.vcf(\\.gz)?$")
assert_file(POPMAP_HAP, "PopMap haploide", "popmap.*hap.*\\.txt$")
assert_file(POPMAP_DIP, "PopMap diploide", "popmap.*dip.*\\.txt$")
assert_file(BLOCKS_FILE, "Tabla de bloques diemPy")
assert_file(ZTCN_ASSIGN_HAP, "Asignaciones ZTCN haploides")
assert_file(ZTCN_ASSIGN_DIP, "Asignaciones ZTCN diploides")

required_packages <- c("tidypopgen", "dplyr", "tibble")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0L) {
  abort_run("Paquetes R faltantes: ", paste(missing_packages, collapse = ", "))
}

actual_version <- as.character(utils::packageVersion("tidypopgen"))
if (actual_version != REQUIRED_TIDYPOPGEN_VERSION) {
  abort_run(
    "Versión de tidypopgen distinta de la requerida. Encontrada = ", actual_version,
    "; requerida = ", REQUIRED_TIDYPOPGEN_VERSION
  )
}
log_msg("tidypopgen version: ", actual_version)

parameters <- data.frame(
  parameter = c(
    "base_dir", "vcf_hap", "vcf_dip", "popmap_hap", "popmap_dip",
    "blocks_file", "ztcn_assign_hap", "ztcn_assign_dip",
    "tidypopgen_version", "fst_method", "background_definition",
    "cross_ploidy_design", "region_sets", "individual_scope", "haploid_handling", "diploid_handling",
    "min_loci_region", "min_individuals_per_population", "n_cores",
    "negative_fst_handling", "priority_classes"
  ),
  value = c(
    BASE_DIR, VCF_HAP, VCF_DIP, POPMAP_HAP, POPMAP_DIP,
    BLOCKS_FILE, ZTCN_ASSIGN_HAP, ZTCN_ASSIGN_DIP,
    actual_version, "Hudson", "same contig excluding only focal analysis interval",
    "every candidate interval evaluated independently in HAP and DIP",
    "EXACT_WINDOWS and HARMONIZED_REGIONS",
    "all individuals in each population", "gt_pseudohaploid(test_n_loci=NULL)",
    "diploid dosage 0/1/2", as.character(MIN_LOCI_REGION),
    as.character(MIN_INDIVIDUALS_PER_POP), as.character(N_CORES),
    "preserved", if (is.null(PRIORITY_CLASSES)) "ALL" else paste(PRIORITY_CLASSES, collapse = ",")
  ),
  stringsAsFactors = FALSE
)
write_tsv(parameters, file.path(OUTDIR, "run_parameters.tsv"))


blocks_all <- add_comparison_columns(read_blocks(BLOCKS_FILE))
exact_windows <- build_exact_windows(blocks_all)
harmonized_regions <- build_harmonized_regions(blocks_all)

write_tsv(blocks_all, file.path(OUTDIR, "blocks_input_audit.tsv"))
write_tsv(exact_windows, file.path(OUTDIR, "analysis_regions_EXACT_WINDOWS.tsv"))
write_tsv(harmonized_regions, file.path(OUTDIR, "analysis_regions_HARMONIZED_REGIONS.tsv"))

log_msg(
  "Bloques fuente = ", nrow(blocks_all),
  "; ventanas exactas = ", nrow(exact_windows),
  "; regiones armonizadas = ", nrow(harmonized_regions),
  "; HAP-only = ", sum(harmonized_regions$detection_class == "HAP_only"),
  "; DIP-only = ", sum(harmonized_regions$detection_class == "DIP_only"),
  "; HAP+DIP = ", sum(harmonized_regions$detection_class == "HAP_and_DIP")
)

popmap_hap_raw <- read_popmap(POPMAP_HAP, "HAP")
popmap_dip_raw <- read_popmap(POPMAP_DIP, "DIP")
assign_hap <- read_ztcn_assignments(ZTCN_ASSIGN_HAP, "HAP")
assign_dip <- read_ztcn_assignments(ZTCN_ASSIGN_DIP, "DIP")

validate_ztcn_assignment_table(popmap_hap_raw, assign_hap, "HAP")
validate_ztcn_assignment_table(popmap_dip_raw, assign_dip, "DIP")

popmap_hap <- apply_ztcn_reassignment(popmap_hap_raw, assign_hap, "HAP")
popmap_dip <- apply_ztcn_reassignment(popmap_dip_raw, assign_dip, "DIP")

write_tsv(popmap_hap, file.path(OUTDIR, "population_reassignment_audit_HAP.tsv"))
write_tsv(popmap_dip, file.path(OUTDIR, "population_reassignment_audit_DIP.tsv"))

count_audit <- rbind(
  population_count_audit(popmap_hap, "HAP"),
  population_count_audit(popmap_dip, "DIP")
)
write_tsv(count_audit, file.path(OUTDIR, "population_counts_before_after.tsv"))

log_msg(
  "ZTCN HAP: QCZ-like = ", sum(popmap_hap$reassignment_action == "ZTCN_QCZ-like_to_QCZ"),
  "; RIT-like = ", sum(popmap_hap$reassignment_action == "ZTCN_RIT-like_to_RIT"),
  "; excluded = ", sum(popmap_hap$reassignment_action == "ZTCN_excluded")
)
log_msg(
  "ZTCN DIP: QCZ-like = ", sum(popmap_dip$reassignment_action == "ZTCN_QCZ-like_to_QCZ"),
  "; RIT-like = ", sum(popmap_dip$reassignment_action == "ZTCN_RIT-like_to_RIT"),
  "; excluded = ", sum(popmap_dip$reassignment_action == "ZTCN_excluded")
)

gt_hap <- prepare_gen_tibble(VCF_HAP, popmap_hap, "HAP")
gt_dip <- prepare_gen_tibble(VCF_DIP, popmap_dip, "DIP")

# 1) Todas las coordenadas originales, sin fusionar: 35 ventanas x 2 ploidías.
exact_hap <- process_regions(gt_hap, exact_windows, "HAP", "EXACT_WINDOWS")
exact_dip <- process_regions(gt_dip, exact_windows, "DIP", "EXACT_WINDOWS")
exact_long <- dplyr::bind_rows(exact_hap, exact_dip)
exact_wide <- make_wide_results(exact_windows, exact_hap, exact_dip)
write_tsv(exact_long, file.path(OUTDIR, "FST_Hudson_EXACT_WINDOWS_LONG.tsv"))
write_tsv(exact_wide, file.path(OUTDIR, "FST_Hudson_EXACT_WINDOWS_WIDE.tsv"))

# 2) Regiones HAP/DIP solapadas armonizadas por unión: 29 regiones x 2 ploidías.
harmonized_hap <- process_regions(gt_hap, harmonized_regions, "HAP", "HARMONIZED_REGIONS")
harmonized_dip <- process_regions(gt_dip, harmonized_regions, "DIP", "HARMONIZED_REGIONS")
harmonized_long <- dplyr::bind_rows(harmonized_hap, harmonized_dip)
harmonized_wide <- make_wide_results(harmonized_regions, harmonized_hap, harmonized_dip)
write_tsv(harmonized_long, file.path(OUTDIR, "FST_Hudson_HARMONIZED_REGIONS_LONG.tsv"))
write_tsv(harmonized_wide, file.path(OUTDIR, "FST_Hudson_HARMONIZED_REGIONS_WIDE.tsv"))

write_aligned_dumbbell_plot(
  harmonized_wide,
  file.path(OUTDIR, "FST_Hudson_HARMONIZED_REGIONS_HAP_DIP_aligned.png")
)

candidate_summary <- as.data.frame(table(harmonized_regions$detection_class), stringsAsFactors = FALSE)
names(candidate_summary) <- c("detection_class", "n_harmonized_regions")
write_tsv(candidate_summary, file.path(OUTDIR, "candidate_detection_summary.tsv"))

phase_summary <- as.data.frame(table(harmonized_wide$phase_pattern, useNA = "ifany"), stringsAsFactors = FALSE)
names(phase_summary) <- c("phase_pattern", "n_harmonized_regions")
write_tsv(phase_summary, file.path(OUTDIR, "fst_phase_pattern_summary.tsv"))

writeLines(capture.output(utils::sessionInfo()), file.path(OUTDIR, "sessionInfo.txt"))

ok_exact_hap <- sum(exact_hap$block_status == "OK" & exact_hap$background_status == "OK")
ok_exact_dip <- sum(exact_dip$block_status == "OK" & exact_dip$background_status == "OK")
ok_harm_hap <- sum(harmonized_hap$block_status == "OK" & harmonized_hap$background_status == "OK")
ok_harm_dip <- sum(harmonized_dip$block_status == "OK" & harmonized_dip$background_status == "OK")
log_msg(
  "Resultados completos EXACT: HAP = ", ok_exact_hap, "/", nrow(exact_hap),
  "; DIP = ", ok_exact_dip, "/", nrow(exact_dip)
)
log_msg(
  "Resultados completos HARMONIZED: HAP = ", ok_harm_hap, "/", nrow(harmonized_hap),
  "; DIP = ", ok_harm_dip, "/", nrow(harmonized_dip)
)
log_msg("FINISHED SUCCESSFULLY")
cat("FINISHED SUCCESSFULLY\n", file = LOG_FILE, append = TRUE)
