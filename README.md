# Hudson FST cruzado: mismas regiones en haploides y diploides

## Objetivo

Este paquete implementa la comparación solicitada: **cada región candidata de diemPy se calcula en HAP y en DIP usando exactamente las mismas coordenadas**, aunque diemPy la haya detectado originalmente en una sola ploidía.

Para cada región y ploidía se calcula:

- `fst_block_Hudson`: Hudson FST dentro de la región focal.
- `fst_contig_background_Hudson`: Hudson FST en el mismo contig, excluyendo solamente la región focal.
- `delta_fst_block_minus_background = fst_block_Hudson - fst_contig_background_Hudson`.

Se utilizan todos los individuos de las dos poblaciones del contraste local; no se restringe a carriers. Los valores negativos de FST se conservan.

## Dos conjuntos de análisis

### 1. EXACT_WINDOWS

Conserva las 35 coordenadas originales de la tabla de candidatos. Cada una se calcula en HAP y DIP. Produce 70 filas en formato largo.

Este conjunto evita modificar las fronteras originales y sirve como auditoría/sensibilidad a los límites de los bloques.

### 2. HARMONIZED_REGIONS

Fusiona únicamente intervalos que se solapan en el mismo contig y para la misma comparación poblacional. La coordenada común es la unión del intervalo HAP y DIP. Con los insumos incluidos se obtienen 29 regiones:

- 15 detectadas solo en HAP.
- 8 detectadas solo en DIP.
- 6 detectadas en ambas ploidías.

Este conjunto es el recomendado para la figura principal porque cada fila representa una sola región genómica y HAP/DIP quedan perfectamente alineados.

## Comparaciones poblacionales

- QCZ–RIT para candidatos de QCZ/RIT/ZTCN-like.
- LLI–CVG para candidatos de LLI/CVG.
- LIL–LOB para candidatos de LIL/LOB.

La reasignación validada de ZTCN se mantiene sin cambios respecto de la versión 3.

## Ejecución

Descomprimir dentro del directorio de análisis y ejecutar:

```bash
cd /media/server/a77f75fe-fd07-402e-84d7-a7341c29141c1/fertorres/segundo_capitulo/integration_analyses/Julio
unzip fst_hudson_cross_ploidy_all_windows_tidypopgen_package_v4.zip
bash fst_hudson_cross_ploidy_all_windows_tidypopgen_package_v4/run_fst_hudson_cross_ploidy.sh
```

Para cambiar el número de núcleos:

```bash
FST_N_CORES=20 bash fst_hudson_cross_ploidy_all_windows_tidypopgen_package_v4/run_fst_hudson_cross_ploidy.sh
```

## Insumo externo requerido

```text
/media/server/a77f75fe-fd07-402e-84d7-a7341c29141c1/fertorres/segundo_capitulo/integration_analyses/Julio/vcf_maf_md_100kpb_contigs_pseudodip.vcf.gz
```

## Requisitos

- Rscript.
- tidypopgen 0.4.4.
- dplyr.
- tibble.

## Salidas principales

Se crea `FST_Hudson_cross_ploidy_all_windows_tidypopgen/run_YYYYMMDD_HHMMSS/`.

- `analysis_regions_EXACT_WINDOWS.tsv`
- `FST_Hudson_EXACT_WINDOWS_LONG.tsv`
- `FST_Hudson_EXACT_WINDOWS_WIDE.tsv`
- `analysis_regions_HARMONIZED_REGIONS.tsv`
- `FST_Hudson_HARMONIZED_REGIONS_LONG.tsv`
- `FST_Hudson_HARMONIZED_REGIONS_WIDE.tsv`
- `FST_Hudson_HARMONIZED_REGIONS_HAP_DIP_aligned.png`
- `candidate_detection_summary.tsv`
- `fst_phase_pattern_summary.tsv`
- archivos de auditoría, parámetros, sesión y log.

## Interpretación

- `delta < 0`: la región tiene menor diferenciación que el resto de su contig.
- `delta ≈ 0`: no existe una desviación local clara respecto del contig.
- `delta > 0`: la región está más diferenciada que el resto del contig.

La comparación es descriptiva. No es por sí sola una prueba formal de introgresión ni de significancia. Para una prueba formal se requiere un fondo nulo empírico con segmentos del mismo contig igualados por longitud, número de SNPs y, idealmente, propiedades de diversidad/missingness.
