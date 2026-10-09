"""
Pipeline: Experiment on REAL workflow benchmarks.

Runs HEFT, baseline QPSO, static-partition QPSO, original ACMR,
ACMR-Soft (redesigned), and GA on 25 real scientific workflow instances
(1000Genome, Cycles, Epigenomics, Montage, Seismology, SoyKB, sraSearch --
real DAG structure + real measured runtimes from Chameleon Cloud executions,
via WfCommons/WfInstances), using the DAG-generalized scheduler in
dag_core.py.

Protocol: each of the 25 real workflow files is ONE paired trial (all
algorithms scheduled on the exact same workflow, same random seed where
applicable) -- this mirrors how the workflow-scheduling literature actually
evaluates these benchmarks (one workflow instance scheduled across the
target platform), rather than stacking many large workflows onto the
paper's small 4-processor testbed at once (which we tested and found
produces contention outside the model's designed operating range -- see
REAL_DATA_NOTES.md).

Usage:
    python3 pipeline_real_data.py --filelist real_workflow_filelist.txt
"""

import argparse
import json
import numpy as np
from scipy import stats

import dag_core as dc
from ga_baseline_dag import run_ga_dag

METRICS = ['energy', 'reliability', 'load_imbalance', 'violations', 'f_obj']
ALGOS = ['HEFT', 'baseline', 'static', 'acmr', 'acmr_soft', 'GA']


def load_filelist(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def run_pipeline(filelist_path, seed0=1000, bias_strength=0.10):
    files = load_filelist(filelist_path)
    results = {a: {m: [] for m in METRICS} for a in ALGOS}
    meta = []

    for trial, path in enumerate(files):
        dag = dc.load_real_workflow(path)
        deadline = np.array([dc.deadline_for(dag)])
        tasks = [dag]
        seed = seed0 + trial

        h = dc.heft_schedule_dag(tasks, deadline)
        for m in METRICS:
            results['HEFT'][m].append(h[m])
        seed_particle = dc.heft_seed_particle_dag(tasks, h['assign_proc'], h['assign_mode'])

        for variant, key in [('baseline', 'baseline'), ('static', 'static'), ('acmr', 'acmr')]:
            rng_alg = np.random.default_rng(seed + 100000)
            r = dc.run_qpso_dag(tasks, deadline, rng_alg, variant=variant, seed_particle=seed_particle)
            for m in METRICS:
                results[key][m].append(r[m])

        rng_alg = np.random.default_rng(seed + 100000)
        r = dc.run_qpso_dag(tasks, deadline, rng_alg, variant='acmr_soft',
                             seed_particle=seed_particle, bias_strength=bias_strength)
        for m in METRICS:
            results['acmr_soft'][m].append(r[m])

        rng_ga = np.random.default_rng(seed + 100000)
        g = run_ga_dag(tasks, deadline, rng_ga)
        for m in METRICS:
            results['GA'][m].append(g[m])

        meta.append(dict(file=path, n_modules=len(dag), deadline=float(deadline[0])))
        print(f"trial {trial+1}/{len(files)}: {path.split('/')[-1]:50s} "
              f"({len(dag)} modules) done", flush=True)

    return results, meta


def holm_bonferroni(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    corrected = np.empty(m)
    for rank, i in enumerate(idx):
        corrected[i] = pvals[i] * (m - rank)
    sorted_corrected = np.maximum.accumulate(corrected[idx])
    out = np.empty(m)
    out[idx] = np.minimum(sorted_corrected, 1.0)
    return out


def cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def analyze(results):
    summary = {a: {m: (float(np.mean(results[a][m])), float(np.std(results[a][m])))
                    for m in METRICS} for a in ALGOS}

    comparisons, raw_p = [], []
    for other in [x for x in ALGOS if x != 'acmr_soft']:
        for m in METRICS:
            a_vals = np.array(results['acmr_soft'][m])
            b_vals = np.array(results[other][m])
            t_stat, p_t = stats.ttest_rel(a_vals, b_vals)
            try:
                w_stat, p_w = stats.wilcoxon(a_vals, b_vals)
            except ValueError:
                p_w = np.nan
            d = cohens_d_paired(a_vals, b_vals)
            pct = 100 * (a_vals.mean() - b_vals.mean()) / (abs(b_vals.mean()) + 1e-12)
            comparisons.append(dict(metric=m, vs=other, pct_diff=pct, t_p=p_t, wilcoxon_p=p_w, cohens_d=d))
            raw_p.append(p_t)

    corrected = holm_bonferroni(np.array(raw_p))
    for c, p_adj in zip(comparisons, corrected):
        c['t_p_holm'] = float(p_adj)
        c['sig_holm_0.05'] = bool(p_adj < 0.05)
    return summary, comparisons


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='real_workflow_filelist.txt')
    args = parser.parse_args()

    results, meta = run_pipeline(args.filelist)
    summary, comparisons = analyze(results)

    with open('/home/claude/acmr_experiment/real_data_raw.json', 'w') as f:
        json.dump(dict(results=results, meta=meta), f, indent=2)
    with open('/home/claude/acmr_experiment/real_data_summary.json', 'w') as f:
        json.dump(dict(summary=summary, comparisons=comparisons), f, indent=2)

    print("\n===== SUMMARY (mean +/- sd, n=%d real workflows) =====" % len(meta))
    header = f"{'Algorithm':12s}" + "".join(f"{m:>16s}" for m in METRICS)
    print(header)
    for a in ALGOS:
        row = f"{a:12s}"
        for m in METRICS:
            mean, sd = summary[a][m]
            row += f"{mean:9.2f}+-{sd:5.2f} "
        print(row)

    print("\n===== ACMR-Soft vs OTHERS (Holm-corrected paired t-test, Cohen's d) =====")
    for c in comparisons:
        sig = "*" if c['sig_holm_0.05'] else " "
        print(f"{sig} acmr_soft vs {c['vs']:10s} [{c['metric']:15s}] "
              f"pct_diff={c['pct_diff']:+8.1f}%  t_p={c['t_p']:.2e}  holm_p={c['t_p_holm']:.2e}  d={c['cohens_d']:+.2f}")
