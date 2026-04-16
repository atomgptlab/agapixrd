# AGAPI-XRD: Reproducibility Repository

Reproducibility code, benchmark scripts, and figure-generation pipelines for the AGAPI-XRD study — a hybrid framework for automated crystal-structure identification from powder X-ray diffraction (XRD) data.

---

## Abstract

X-ray diffraction (XRD) remains one of the most powerful experimental techniques for characterizing materials, yet the path from raw diffraction data to a refined atomic structure continues to demand significant domain expertise and manual intervention. We present AGAPI-XRD, a hybrid computational framework that integrates DiffractGPT (a generative pretrained transformer trained on thousands of crystal structures and their simulated XRD patterns), elemental and stoichiometric pattern matching against the JARVIS-DFT and COD materials databases, optional ALIGNN-FF geometry relaxation, and classical Rietveld refinement into a unified, accessible API hosted on the AtomGPT.org API (AGAPI) platform.

The combined pipeline is exposed through the AGAPI interface at `https://atomgpt.org/xrd`, enabling seamless programmatic access for experimentalists, high-throughput screening workflows, and integration with broader agentic AI frameworks for materials discovery. We benchmark the approach on 276 minerals from the RRUFF powder XRD database, achieving 96.7% structure-identification coverage with lattice-parameter mean absolute errors of approximately 1.1 Å for unit-cell lengths and positive skill scores across lattice dimensions and volume.

---

## Repository Structure

```
agapi_xrd_paper/
├── src/
│   ├── agapi_xrd_scripts/        # Benchmark drivers and result compilation
│   │   ├── rruff_xrd_analysis.py
│   │   ├── analyse_filtered_rruff.py  (symlink → plotting_scripts/)
│   │   ├── compile_agapi_replication_metrics.py
│   │   ├── compile_results.sh
│   │   └── xrd_simulate_alex_pbe_hull.py
│   ├── slurm_scripts/            # HPC job submission files
│   │   ├── xrd_pipeline_none.job
│   │   ├── xrd_pipeline_none_alignnff.job
│   │   ├── xrd_pipeline_gsas2.job
│   │   ├── xrd_pipeline_bmgn.job
│   │   ├── xrd_pipeline_bmgn_alignnff.job
│   │   └── xrd_simulate_dataset.job
│   ├── plotting_scripts/         # Figure generation
│   │   ├── analyse_filtered_rruff.py
│   │   ├── analyse_filtered_rruff_runner.sh
│   │   ├── plot_refinement_mae.py
│   │   ├── plot_refinement_jsd_18panel.py
│   │   ├── plot_refinement_mae_bmgn_vs_bmgn_alignnff.py
│   │   ├── plot_match_rate_crystal_systems.py
│   │   ├── plot_match_rate.py
│   │   ├── rruff_stoich_pie.py
│   │   └── stoich.sh
│   └── generate_figures.sh       # Top-level figure orchestration script
├── runs/                         # Output directory (populated after benchmark runs)
│   ├── no_refinement/
│   ├── no_refinement_alignnff/
│   ├── gsas2/
│   ├── bmgn/
│   └── bmgn_alignnff/
├── testing/                      # Unit and integration test assets
├── key.txt                       # AtomGPT.org API key (user-provided, not tracked)
└── README.md
```

---

## Installation

This project is part of the JARVIS ecosystem. Create and activate a minimal conda environment:

```bash
conda create -n jarvis python=3.10 -y
conda activate jarvis
pip install --upgrade pip
pip install jarvis-tools
```

### API Key

Obtain an API key from [atomgpt.org](https://atomgpt.org) and write it to a `key.txt` file at the repository root. The benchmark scripts read this file at runtime.

```bash
echo "YOUR_ATOMGPT_API_KEY" > key.txt
```

---

## Reproducing the Benchmark

Benchmark runs are submitted as Slurm jobs from `src/slurm_scripts/`. The five pipeline variants correspond to different refinement backends and optional ALIGNN-FF pre-relaxation:

| Job file | Refinement backend | ALIGNN-FF pre-relaxation |
|---|---|---|
| `xrd_pipeline_none.job` | None | No |
| `xrd_pipeline_none_alignnff.job` | None | Yes |
| `xrd_pipeline_gsas2.job` | GSAS-II | No |
| `xrd_pipeline_bmgn.job` | BGMN | No |
| `xrd_pipeline_bmgn_alignnff.job` | BGMN | Yes |

Submit any or all jobs from the repository root:

```bash
sbatch src/slurm_scripts/xrd_pipeline_none.job
sbatch src/slurm_scripts/xrd_pipeline_none_alignnff.job
sbatch src/slurm_scripts/xrd_pipeline_gsas2.job
sbatch src/slurm_scripts/xrd_pipeline_bmgn.job
sbatch src/slurm_scripts/xrd_pipeline_bmgn_alignnff.job
```

Job outputs are written to the corresponding subdirectory under `runs/`.

---

## Analysis and Figure Generation

After the Slurm jobs complete, run the following steps in order.

### 1. Analyze benchmark predictions

```bash
bash src/plotting_scripts/analyse_filtered_rruff_runner.sh
```

Processes each `runs/` subdirectory and computes benchmark statistics against the filtered RRUFF ground-truth lattice parameters (`a, b, c ≤ 10 Å`).

### 2. Generate manuscript figures

```bash
bash src/generate_figures.sh
```

Regenerates all figure assets from the processed run outputs.

### 3. Aggregate run-level results

```bash
bash src/agapi_xrd_scripts/compile_results.sh
```

Consolidates outputs from all run directories into a unified summary.

### 4. Compile replication metrics

```bash
python src/agapi_xrd_scripts/compile_agapi_replication_metrics.py
```

Produces `replication_summary.json`, a structured metrics file suitable for reviewer inspection.

---

## Script Reference

### Benchmark and analysis

| Script | Description |
|---|---|
| `agapi_xrd_scripts/rruff_xrd_analysis.py` | Primary benchmark driver; queries AGAPI-XRD predictions and writes outputs to `runs/` |
| `plotting_scripts/analyse_filtered_rruff.py` | Evaluates prediction outputs against filtered RRUFF ground-truth lattice parameters |
| `plotting_scripts/analyse_filtered_rruff_runner.sh` | Shell wrapper for the filtered RRUFF analysis workflow |
| `agapi_xrd_scripts/compile_agapi_replication_metrics.py` | Compiles benchmark outputs into `replication_summary.json` |
| `agapi_xrd_scripts/compile_results.sh` | Aggregates results from multiple run directories |
| `agapi_xrd_scripts/xrd_simulate_alex_pbe_hull.py` | Standalone dense XRD simulator for dataset generation |

### Figure generation

| Script | Description |
|---|---|
| `plotting_scripts/plot_refinement_mae.py` | MAE comparison across six lattice parameters for all refinement settings |
| `plotting_scripts/plot_refinement_jsd_18panel.py` | 18-panel Jensen–Shannon divergence figure across lattice parameters and refinement settings |
| `plotting_scripts/plot_match_rate_crystal_systems.py` | Crystal-system histograms and pattern-matching summaries |
| `plotting_scripts/plot_refinement_mae_bmgn_vs_bmgn_alignnff.py` | Lattice-parameter MAE comparison between BGMN and BGMN + ALIGNN-FF workflows |
| `plotting_scripts/plot_match_rate.py` | Pattern-matching rate summaries |
| `plotting_scripts/rruff_stoich_pie.py` | Elemental and stoichiometric composition visualizations for the RRUFF benchmark set |
| `plotting_scripts/stoich.sh` | Shell helper for the stoichiometry figure workflow |

---

## Expected Outputs

A complete run produces the following artifacts under `runs/` and the repository root:

- Per-condition benchmark outputs in `runs/{condition}/`
- Lattice-parameter MAE comparison figures (PNG)
- 18-panel Jensen–Shannon divergence figure (PNG)
- Crystal-system and pattern-matching histograms (PNG)
- Stoichiometric composition plots (PNG)
- `replication_summary.json` — structured replication metrics for all conditions

---

## Notes

**Filename conventions.** Several scripts and job files use the string `bmgn`; the manuscript refers to the BGMN refinement engine. These names are preserved verbatim to maintain correspondence between the code and the paper.

**Benchmark scope.** The manuscript benchmark is restricted to RRUFF entries with `a, b, c ≤ 10 Å` to enable consistent lattice-parameter comparison across all tested structures.

**External dependencies.** The full pipeline interfaces with the AtomGPT.org AGAPI platform, the JARVIS and COD reference databases, GSAS-II, BGMN, and (optionally) ALIGNN-FF. Ensure these services and tools are accessible in your compute environment before running the benchmark jobs.

---

## Citation

If you use this code or data, please cite the AGAPI-XRD manuscript and the relevant JARVIS / AtomGPT ecosystem resources. Citation details will be updated upon publication.

---

## Contact

For questions regarding the workflow, benchmark, or manuscript, contact the corresponding authors listed in the paper.
