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
├── manuscript.tex                     # LaTeX source for the paper
├── scripts/
│   ├── benchmark/                     # Scripts that call the AGAPI-XRD API and produce raw results
│   │   ├── rruff_xrd_analysis.py      # Benchmark driver for the RRUFF mineral dataset
│   │   ├── alex_xrd_analysis.py       # Benchmark driver for the Alexandria PBE-hull dataset
│   │   ├── xrd_simulate_alex_pbe_hull.py  # Simulate XRD patterns for the Alexandria dataset
│   │   ├── compile_agapi_replication_metrics.py  # Compile outputs into replication_summary.json
│   │   ├── compile_results.sh         # Aggregates results from all run directories
│   │   └── runs_to_compile.txt        # List of run directories to include in compilation
│   ├── analysis/                      # Scripts that evaluate benchmark outputs
│   │   ├── analyse_filtered_rruff.py  # Compute lattice-parameter metrics vs. RRUFF ground truth
│   │   └── analyse_filtered_rruff_runner.sh  # Run analysis across all run directories
│   └── figures/                       # Scripts that generate manuscript figures
│       ├── generate_figures.sh        # Orchestrator: run all figure scripts in order
│       ├── plot_refinement_mae.py     # MAE across six lattice parameters for all refinement settings
│       ├── plot_refinement_jsd_18panel.py  # 18-panel Jensen-Shannon divergence figure
│       ├── plot_match_rate_crystal_systems.py  # Crystal-system histograms and match-rate summaries
│       ├── plot_refinement_mae_bmgn_vs_bmgn_alignnff.py  # BGMN vs. BGMN+ALIGNN-FF MAE comparison
│       ├── plot_match_rate.py         # Pattern-matching rate summaries
│       ├── rruff_stoich_pie.py        # Elemental composition pie chart for the RRUFF benchmark set
│       └── stoich.sh                  # Shell helper for the stoichiometry figure
├── slurm/                             # HPC job submission scripts (submit from repo root)
│   ├── rruff_none.job                 # RRUFF benchmark — no refinement
│   ├── rruff_none_alignnff.job        # RRUFF benchmark — no refinement + ALIGNN-FF
│   ├── rruff_gsas2.job                # RRUFF benchmark — GSAS-II refinement
│   ├── rruff_gsas2_alignnff.job       # RRUFF benchmark — GSAS-II refinement + ALIGNN-FF
│   ├── rruff_bmgn.job                 # RRUFF benchmark — BGMN refinement
│   ├── rruff_bmgn_alignnff.job        # RRUFF benchmark — BGMN refinement + ALIGNN-FF
│   ├── alex_none.job                  # Alexandria benchmark — no refinement
│   ├── alex_none_alignnff.job         # Alexandria benchmark — no refinement + ALIGNN-FF
│   ├── alex_gsas2.job                 # Alexandria benchmark — GSAS-II refinement
│   ├── alex_gsas2_alignnff.job        # Alexandria benchmark — GSAS-II refinement + ALIGNN-FF
│   ├── alex_bmgn.job                  # Alexandria benchmark — BGMN refinement
│   ├── alex_bmgn_alignnff.job         # Alexandria benchmark — BGMN refinement + ALIGNN-FF
│   └── simulate_dataset.job           # Simulate XRD patterns for the Alexandria dataset
├── runs/                              # Output directory (populated after benchmark runs, git-ignored)
├── testing/                           # Test assets
└── key.txt                            # AtomGPT.org API key (user-provided, git-ignored)
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

Benchmark runs are submitted as Slurm jobs from `slurm/`. Two datasets are supported (RRUFF and Alexandria), each with five pipeline variants (refinement backend × ALIGNN-FF pre-relaxation):

| Job file | Dataset | Refinement backend | ALIGNN-FF |
|---|---|---|---|
| `slurm/rruff_none.job` | RRUFF | None | No |
| `slurm/rruff_none_alignnff.job` | RRUFF | None | Yes |
| `slurm/rruff_gsas2.job` | RRUFF | GSAS-II | No |
| `slurm/rruff_gsas2_alignnff.job` | RRUFF | GSAS-II | Yes |
| `slurm/rruff_bmgn.job` | RRUFF | BGMN | No |
| `slurm/rruff_bmgn_alignnff.job` | RRUFF | BGMN | Yes |
| `slurm/alex_none.job` | Alexandria | None | No |
| `slurm/alex_none_alignnff.job` | Alexandria | None | Yes |
| `slurm/alex_gsas2.job` | Alexandria | GSAS-II | No |
| `slurm/alex_gsas2_alignnff.job` | Alexandria | GSAS-II | Yes |
| `slurm/alex_bmgn.job` | Alexandria | BGMN | No |
| `slurm/alex_bmgn_alignnff.job` | Alexandria | BGMN | Yes |

Submit from the repository root:

```bash
sbatch slurm/rruff_none.job
sbatch slurm/rruff_bmgn.job
# ... etc.
```

Job outputs are written to the corresponding subdirectory under `runs/`.

---

## Analysis and Figure Generation

After the Slurm jobs complete, run the following steps in order from the **repository root**.

### 1. Analyze benchmark predictions

```bash
bash scripts/analysis/analyse_filtered_rruff_runner.sh
```

Processes each `runs/` subdirectory and computes lattice-parameter metrics against the filtered RRUFF ground truth (`a, b, c ≤ 10 Å`).

### 2. Generate manuscript figures

```bash
bash scripts/figures/generate_figures.sh
```

Regenerates all figure assets from the processed run outputs.

### 3. Aggregate run-level results

```bash
bash scripts/benchmark/compile_results.sh
```

Consolidates outputs from all run directories into a unified summary.

### 4. Compile replication metrics

```bash
python scripts/benchmark/compile_agapi_replication_metrics.py
```

Produces `replication_summary.json`, a structured metrics file suitable for reviewer inspection.

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
