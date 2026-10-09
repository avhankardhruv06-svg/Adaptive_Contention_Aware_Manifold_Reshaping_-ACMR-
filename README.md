# Python Code: Real-Workflow ACMR/ACMR-Soft Pipeline

Complete, runnable code for the synthetic and real-data experiments.

## Files

| File | Purpose |
|---|---|
| `acmr_core.py` | Synthetic (linear-chain) simulator: Tables I/II, Eq. 4-19, Algorithm 1, plus the `acmr_soft`/`acmr_targeted` redesigns and HEFT-seeding support |
| `ga_baseline.py` | GA baseline for the synthetic model |
| `dag_core.py` | DAG-generalized scheduler, HEFT, and all QPSO/ACMR variants for real (branching) workflows |
| `ga_baseline_dag.py` | GA baseline for the DAG model |
| `select_workflow_files.py` | Selects a balanced set of real workflow instances from a local WfInstances checkout |
| `pipeline_real_data.py` | Runs HEFT/baseline/static/ACMR/ACMR-Soft/GA on the selected real workflows, with paired statistics |
| `visualize_real_data.py` | Generates the 7 comparison figures from the pipeline's output JSON |

## Setup

1. **Get the real workflow traces** (one-time download, ~100MB):
   ```bash
   curl -sL -o wfinstances.tar.gz \
     "https://codeload.github.com/wfcommons/WfInstances/tar.gz/refs/heads/main"
   tar -xzf wfinstances.tar.gz
   ```
   This creates `WfInstances-main/pegasus/{1000genome,cycles,epigenomics,montage,seismology,soykb,srasearch}/*.json`.

2. **Install dependencies**:
   ```bash
   pip install numpy scipy matplotlib --break-system-packages
   ```

## Running the full pipeline

```bash
# 1. Select a balanced set of real workflow instances (60 here; adjust as needed)
python3 select_workflow_files.py \
    --source_dir WfInstances-main/pegasus \
    --target_total 60 \
    --max_modules 700 \
    --out_filelist real_workflow_filelist.txt \
    --copy_to real_workflows

# 2. Run all six algorithms (HEFT, baseline QPSO, static, ACMR, ACMR-Soft, GA)
#    across every selected workflow, with paired statistics
python3 pipeline_real_data.py --filelist real_workflow_filelist_portable.txt

# 3. Generate the 7 comparison figures
python3 visualize_real_data.py
```

Step 2 took ~8 minutes for 60 instances (22-676 modules each) on a single
core; runtime scales roughly with total module count across all selected
instances, since QPSO/GA cost is dominated by per-module decode/schedule
evaluation.

## Adjusting the experiment

- **More/fewer real instances:** change `--target_total` in step 1.
- **Larger workflows:** raise `--max_modules` in step 1 (runtime scales
  accordingly — a single 676-module instance takes ~5s per QPSO/GA call).
- **Different ACMR-Soft bias strength:** `pipeline_real_data.py` takes a
  `bias_strength` argument to `run_pipeline()` (default 0.10, the value
  found best via the synthetic-benchmark sweep — see `acmr_core.py`'s
  `run_qpso` docstring).
- **Synthetic benchmark instead of real data:** use `acmr_core.py` +
  `ga_baseline.py` directly (see `experiment_a_comparison.py`,
  `experiment_c_sensitivity.py`, and `experiment_d_redesign.py` from the
  earlier deliverables for ready-made driver scripts).

## Output

`pipeline_real_data.py` writes:
- `real_data_raw.json` — every metric, every trial, every algorithm
- `real_data_summary.json` — means/SDs plus all pairwise statistics
  (paired t-test, Wilcoxon, Holm-Bonferroni correction, Cohen's d)

`visualize_real_data.py` reads the raw results and writes 7 PNG figures
covering per-metric bar comparisons, full distributions, a per-application
breakdown, and an energy-vs-violations trade-off plot.
