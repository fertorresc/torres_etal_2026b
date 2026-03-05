# Genomic analyses of hybridization and admixture in *Mazzaella laminarioides*

This repository contains the scripts and workflows used in the study:

**"[Title of the manuscript]"**

The aim of this study is to investigate patterns of hybridization, admixture, and population structure among three cryptic lineages of the red alga *Mazzaella laminarioides* distributed along the Chilean coast. Using genome-wide SNP data from 182 individuals (81 haploids and 91 diploids), the analyses combine population genomic and haplotype-based approaches to infer ancestry structure, gene flow, and admixture dynamics across the species’ geographic range.

The repository provides all scripts necessary to reproduce the analyses and figures presented in the manuscript.

---
## Repository structure

```
├── data_processing
│   ├── read_filtering
│   ├── alignment
│   └── variant_calling
│
├── snp_datasets
│   ├── shared_snps
│   ├── combined_haploid_diploid
│   ├── haploid_dataset
│   └── diploid_dataset
│
├── population_structure
│   ├── PCA
│   └── ancestry_inference
│
├── haplotype_analyses
│   ├── phasing
│   ├── chromopainter
│   ├── coancestry_matrices
│   └── ancestry_networks
│
├── admixture_analyses
│   ├── globetrotter
│   └── treemix
│
├── genome_scans
│   └── fst_analyses
│
├── figures
│   └── scripts_for_figure_generation
│
└── README.md
```
---

# Analytical workflow

The analyses performed in this study follow the workflow described below.

### 1. Read processing and alignment

Raw sequencing reads were quality-filtered and aligned to the *Mazzaella laminarioides* reference genome using **BWA-MEM**. Resulting alignments were processed to generate sorted BAM files for variant discovery.

### 2. Variant discovery

Single nucleotide polymorphisms (SNPs) were identified following best-practice pipelines. SNP filtering included:

- genotype quality filtering
- read depth thresholds
- missing data filtering
- minor allele frequency filtering
- removal of artefactual variants

### 3. SNP dataset construction

Multiple SNP datasets were generated from the filtered variant matrix to accommodate different analytical requirements:

- **Genome-wide shared SNP dataset**  
  SNPs shared between haploid and diploid individuals for direct comparison across life-cycle stages.

- **Combined haploid–diploid dataset**  
  Joint dataset used for population structure analyses.

- **Ploidy-specific datasets**  
  Separate SNP matrices for haploid and diploid individuals.

- **Haplotype-based dataset**  
  SNPs located within the 11 largest contigs (>600 kb) used for haplotype-based analyses.

### 4. Population genomic structure

Population structure was investigated using:

- **Principal Component Analysis (PCA)**
- **Ancestry inference using sparse non-negative matrix factorization**

### 5. Haplotype-based analyses

Genotypes were phased using **BEAGLE** and analysed using:

- **ChromoPainter** for haplotype painting
- coancestry matrices to quantify haplotype sharing
- ancestry networks based on haplotype similarity

### 6. Admixture inference

Admixture among populations was analysed using:

- **GLOBETROTTER** to detect and date admixture events
- **TreeMix** to infer historical migration among populations

### 7. Genome-wide differentiation

Genome-wide differentiation was quantified using **FST scans** calculated in sliding windows across the genome.

---

# Data availability

Raw sequencing data are available in the **NCBI Sequence Read Archive (SRA)** under BioProject:
Large intermediate files (VCF files, ChromoPainter outputs, etc.) are not included in this repository due to file size limitations but are available at:
