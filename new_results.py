"""
New real-workflow analyses for the ACMR-Soft paper (Section 9.1 and 9.2).

Input : real_data_60_dataset.csv  (60 instances x 6 algorithms)
Output: dataset_with_decomposition.csv, pairwise_stats.csv, per_application.csv,
        fig8_fobj_decomposition.png, fig9_win_tie_loss.png,
        fig10_overrun_magnitude.png
and prints every number used in Tables 5-8.

Usage:  python3 make_new_results.py [path/to/real_data_60_dataset.csv]
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else 'real_data_60_dataset.csv'
A = ['HEFT', 'baseline', 'static', 'acmr', 'acmr_soft', 'GA']
LABELS = ['HEFT', 'Baseline\nQPSO', 'Static\nPartition', 'ACMR\n(original)', 'ACMR-Soft', 'GA']
COLORS = ['#888888', '#4C72B0', '#DD8452', '#C44E52', '#2E7D32', '#55A868']
M = ['energy', 'reliability', 'load_imbalance', 'violations', 'f_obj']

# Objective constants (Table 1 and acmr_core.py)
ALPHA, BETA_W, GAMMA, LAMBDA_PEN = 0.4, 0.4, 0.2, 50.0
ENERGY_NORM, R_MIN = 1000.0, 0.70

d = pd.read_csv(CSV)
n_inst = d.trial_index.nunique()

# ---------------------------------------------------------------------------
# 1. Exact decomposition of F'_obj  (Table 5, Fig. 8)
# ---------------------------------------------------------------------------
d['base'] = ALPHA * d.energy / ENERGY_NORM + BETA_W * (1 - d.reliability) + GAMMA * d.load_imbalance
d['rel_pen'] = 50 * np.maximum(0, R_MIN - d.reliability)
d['dl_pen'] = d.f_obj - d.base - d.rel_pen          # = LAMBDA_PEN * P_dl
d['overrun'] = d.dl_pen / LAMBDA_PEN                 # P_dl, total normalised overrun
d.to_csv('dataset_with_decomposition.csv', index=False)

g = d.groupby('algorithm')[['base', 'rel_pen', 'dl_pen', 'f_obj']].mean().loc[A]
print('== Table 5: decomposition of F_obj (means) ==')
print(g.round(2).to_string())
print('\nshare of F_obj (%):')
print((g[['base', 'rel_pen', 'dl_pen']].div(g.f_obj, axis=0) * 100).round(1).to_string())
print('instances with R_sys < R_min:', (d.reliability < R_MIN).groupby(d.algorithm).mean().loc[A].to_dict())

# ---------------------------------------------------------------------------
# 2. Overrun on violating instances  (Table 6, Fig. 10)
# ---------------------------------------------------------------------------
v = d[d.violations == 1]
print('\n== Table 6: normalised overrun on violating instances ==')
print(v.groupby('algorithm').overrun.agg(['count', 'mean', 'median', 'max']).loc[A].round(2).to_string())

# ---------------------------------------------------------------------------
# 3. Paired tests, Holm correction, win/tie/loss  (Table 7, Fig. 9)
# ---------------------------------------------------------------------------
def wide(m):
    return d.pivot(index='trial_index', columns='algorithm', values=m)[A]

def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p)
    out = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, p[i] * (m - rank))
        out[i] = min(running, 1.0)
    return out

rows = []
for other in [a for a in A if a != 'acmr_soft']:
    for m in M + ['overrun']:
        a = wide(m)['acmr_soft'].values
        b = wide(m)[other].values
        diff = a - b
        t_p = stats.ttest_rel(a, b).pvalue if np.any(diff != 0) else 1.0
        try:
            w_p = stats.wilcoxon(a, b).pvalue
        except ValueError:
            w_p = np.nan
        dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
        better = diff > 1e-9 if m == 'reliability' else diff < -1e-9
        worse = diff < -1e-9 if m == 'reliability' else diff > 1e-9
        rows.append(dict(vs=other, metric=m, soft_mean=a.mean(), other_mean=b.mean(),
                         pct_diff=100 * (a.mean() - b.mean()) / abs(b.mean()),
                         t_p=t_p, wilcoxon_p=w_p, cohens_d=dz,
                         win=int(better.sum()), tie=int(n_inst - better.sum() - worse.sum()),
                         loss=int(worse.sum())))
r = pd.DataFrame(rows)
primary = r.metric != 'overrun'          # original 25-test family
r.loc[primary, 'holm_p'] = holm(r.loc[primary, 't_p'])
r.loc[~primary, 'holm_p'] = holm(r.loc[~primary, 't_p'])   # separate 5-test overrun family
r.to_csv('pairwise_stats.csv', index=False)
print('\n== Table 7 and pairwise statistics ==')
pd.set_option('display.width', 200)
print(r.round(4).to_string())

# ---------------------------------------------------------------------------
# 4. Friedman test and mean ranks  (Table 8)
# ---------------------------------------------------------------------------
print('\n== Table 8: Friedman test and mean ranks (1 = best) ==')
for m in M + ['overrun']:
    W = wide(m)
    sign = -1 if m == 'reliability' else 1
    ranks = (sign * W).rank(axis=1).mean()
    p = stats.friedmanchisquare(*[W[a] for a in A]).pvalue
    print(f'{m:15s} p={p:.2g}  ' + '  '.join(f'{a}={ranks[a]:.2f}' for a in A))

# Per application means (supporting numbers, e.g. the SoyKB energy remark)
d.groupby(['application', 'algorithm'])[['energy', 'reliability', 'violations', 'f_obj', 'overrun']] \
 .mean().unstack('algorithm').to_csv('per_application.csv')

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Fig. 8: stacked decomposition
fig, ax = plt.subplots(figsize=(7.5, 5))
bottom = np.zeros(len(A))
for col, lab, c in [('base', 'Weighted energy/reliability/imbalance', '#4C72B0'),
                    ('rel_pen', 'Reliability floor penalty', '#DD8452'),
                    ('dl_pen', 'Deadline overrun penalty', '#C44E52')]:
    ax.bar(LABELS, g[col], bottom=bottom, label=lab, color=c, edgecolor='black', lw=.5)
    bottom += g[col].values
for i, val in enumerate(bottom):
    ax.text(i, val + 1, f'{val:.1f}', ha='center', fontsize=9)
ax.set_ylabel("Mean $F'_{obj}$ contribution")
ax.set_title("Decomposition of $F'_{obj}$ (60 real workflows)")
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig('fig8_fobj_decomposition.png', dpi=200); plt.close()

# Fig. 9: win/tie/loss, 2x3 grid
mets = ['energy', 'reliability', 'load_imbalance', 'violations', 'f_obj', 'overrun']
mlab = ['Energy', 'Reliability', 'Load imb.', 'Violations', "$F'_{obj}$", 'Overrun']
others = ['HEFT', 'baseline', 'static', 'acmr', 'GA']
onames = ['HEFT', 'Baseline QPSO', 'Static partition', 'Original ACMR', 'GA']
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5)); axes = axes.ravel()
for ax, o, name in zip(axes, others, onames):
    s = r[r.vs == o].set_index('metric').loc[mets]
    y = np.arange(len(mets))
    ax.barh(y, s.win, color='#2E7D32', label='ACMR-Soft better')
    ax.barh(y, s.tie, left=s.win, color='#BBBBBB', label='Tie')
    ax.barh(y, s.loss, left=s.win + s.tie, color='#C44E52', label='ACMR-Soft worse')
    for yy, (w, t, l) in enumerate(zip(s.win, s.tie, s.loss)):
        for x0, val, txtc in [(0, w, 'white'), (w, t, 'black'), (w + t, l, 'white')]:
            if val >= 5:
                ax.text(x0 + val / 2, yy, str(val), ha='center', va='center', fontsize=8, color=txtc)
    ax.set_yticks(y); ax.set_yticklabels(mlab, fontsize=9)
    ax.set_title(f'vs {name}', fontsize=11); ax.set_xlim(0, n_inst)
    ax.invert_yaxis(); ax.set_xlabel('Instances', fontsize=9)
h, l = axes[0].get_legend_handles_labels()
axes[5].axis('off'); axes[5].legend(h, l, loc='center', fontsize=11, frameon=False)
fig.suptitle(f'Per-instance win/tie/loss of ACMR-Soft (n = {n_inst} paired real instances)', fontsize=12)
plt.tight_layout(); plt.savefig('fig9_win_tie_loss.png', dpi=200); plt.close()

# Fig. 10: overrun on violating instances
fig, ax = plt.subplots(figsize=(7.5, 4.5))
bp = ax.boxplot([v[v.algorithm == a].overrun for a in A], tick_labels=LABELS, showmeans=True,
                patch_artist=True,
                meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6),
                medianprops=dict(color='black'))
for patch, c in zip(bp['boxes'], COLORS):
    patch.set_facecolor(c); patch.set_alpha(.7)
ax.set_ylabel('Normalized overrun $(C-D)/D$')
ax.set_title('Overrun magnitude on deadline-violating instances')
ax.tick_params(axis='x', labelsize=8)
plt.tight_layout(); plt.savefig('fig10_overrun_magnitude.png', dpi=200); plt.close()

print('\nWrote fig8_fobj_decomposition.png, fig9_win_tie_loss.png, fig10_overrun_magnitude.png')
