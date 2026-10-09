"""
Visualization suite for the real-workflow pipeline results.

Produces 7 figures comparing ACMR (original) and ACMR-Soft against HEFT,
baseline QPSO, static partition, and GA on 60 real workflow execution
traces (1000Genome, Cycles, Epigenomics, Montage, Seismology, SoyKB,
sraSearch).

Figures:
  1. fig1_violations_bar.png       - Deadline violations, all 6 algorithms
  2. fig2_energy_bar.png           - Energy consumption, all 6 algorithms
  3. fig3_reliability_bar.png      - System reliability, all 6 algorithms
  4. fig4_fobj_bar.png             - Composite objective, all 6 algorithms
  5. fig5_distributions_box.png    - Box plots: violations & f_obj spread
  6. fig6_per_application.png      - ACMR vs ACMR-Soft, broken down by
                                      real application (1000Genome, Montage, ...)
  7. fig7_pareto_scatter.png       - Energy vs. violations Pareto scatter,
                                      all algorithms, all 60 real instances
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('/home/claude/acmr_experiment/real_data_60_raw.json') as f:
    d = json.load(f)

results = d['results']
meta = d['meta']
ALGOS = ['HEFT', 'baseline', 'static', 'acmr', 'acmr_soft', 'GA']
LABELS = ['HEFT', 'Baseline\nQPSO', 'Static\nPartition', 'ACMR\n(original)',
          'ACMR-Soft\n(redesigned)', 'GA']
COLORS = ['#888888', '#4C72B0', '#DD8452', '#C44E52', '#2E7D32', '#55A868']
METRICS = ['energy', 'reliability', 'load_imbalance', 'violations', 'f_obj']

apps = [m['file'].split('/')[-2] for m in meta]
unique_apps = sorted(set(apps))


def bar_chart(metric, title, ylabel, filename, highlight_lower_better=True):
    means = [np.mean(results[a][metric]) for a in ALGOS]
    sds = [np.std(results[a][metric]) for a in ALGOS]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(LABELS, means, yerr=sds, capsize=5, color=COLORS,
                   edgecolor='black', linewidth=0.6)

    # highlight the best-performing bar
    best_idx = int(np.argmin(means)) if highlight_lower_better else int(np.argmax(means))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(2.5)

    # annotate ACMR and ACMR-Soft specifically for easy visual comparison
    for idx, name in [(3, 'ACMR'), (4, 'ACMR-Soft')]:
        ax.annotate(f'{means[idx]:.2f}', xy=(idx, means[idx] + sds[idx]),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', labelsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f'saved {filename}')


# ---- Figures 1-4: per-metric bar comparisons across all six algorithms ----
bar_chart('violations', 'Real Workflows: Deadline Violations\n(60 real instances, mean \u00b1 SD)',
          'Fraction of workflows violating deadline', 'fig1_violations_bar.png')

bar_chart('energy', 'Real Workflows: Energy Consumption\n(60 real instances, mean \u00b1 SD)',
          'Energy (model units)', 'fig2_energy_bar.png')

bar_chart('reliability', 'Real Workflows: System Reliability\n(60 real instances, mean \u00b1 SD)',
          'System reliability $R_{sys}$', 'fig3_reliability_bar.png', highlight_lower_better=False)

bar_chart('f_obj', "Real Workflows: Composite Objective $F'_{obj}$\n(60 real instances, mean \u00b1 SD)",
          "$F'_{obj}$ (lower is better)", 'fig4_fobj_bar.png')


# ---- Figure 5: distribution views. Violations is binary (0/1) per real
# instance, so a boxplot isn't informative there -- a jittered strip plot
# showing every instance is; F'_obj is continuous, so a boxplot is fine. ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

rng = np.random.default_rng(0)
for i, (algo, color) in enumerate(zip(ALGOS, COLORS)):
    y = np.array(results[algo]['violations'], dtype=float)
    x = np.full_like(y, i) + rng.uniform(-0.12, 0.12, size=len(y))
    axes[0].scatter(x, y, color=color, alpha=0.6, s=45, edgecolor='black', linewidth=0.4)
    axes[0].scatter(i, y.mean(), color=color, marker='D', s=140, edgecolor='black',
                     linewidth=1.3, zorder=5)
axes[0].set_xticks(range(len(ALGOS)))
axes[0].set_xticklabels(LABELS, fontsize=8, rotation=15)
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(['met deadline', 'violated'])
axes[0].set_title('Deadline Violations (every instance;\ndiamond = mean violation rate)')
axes[0].set_ylim(-0.15, 1.15)

data = [results[a]['f_obj'] for a in ALGOS]
bp = axes[1].boxplot(data, tick_labels=LABELS, patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6))
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title("Composite $F'_{obj}$ (distribution)")
axes[1].tick_params(axis='x', labelsize=8, rotation=15)

fig.suptitle('Real Workflows: Full Distribution Across 60 Instances', fontsize=13)
plt.tight_layout()
plt.savefig('fig5_distributions_box.png', dpi=150)
plt.close(fig)
print('saved fig5_distributions_box.png')


# ---- Figure 6: ACMR vs ACMR-Soft broken down by real application ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

app_data_viol = {app: {'acmr': [], 'acmr_soft': [], 'static': []} for app in unique_apps}
app_data_fobj = {app: {'acmr': [], 'acmr_soft': [], 'static': []} for app in unique_apps}
for i, app in enumerate(apps):
    for key in ['acmr', 'acmr_soft', 'static']:
        app_data_viol[app][key].append(results[key]['violations'][i])
        app_data_fobj[app][key].append(results[key]['f_obj'][i])

x = np.arange(len(unique_apps))
width = 0.25

for ax, app_data, title, ylabel in [
    (axes[0], app_data_viol, 'Deadline Violations by Application', 'Mean violations'),
    (axes[1], app_data_fobj, "Composite $F'_{obj}$ by Application", "Mean $F'_{obj}$"),
]:
    acmr_means = [np.mean(app_data[a]['acmr']) for a in unique_apps]
    soft_means = [np.mean(app_data[a]['acmr_soft']) for a in unique_apps]
    static_means = [np.mean(app_data[a]['static']) for a in unique_apps]

    ax.bar(x - width, acmr_means, width, label='ACMR (original)', color='#C44E52', edgecolor='black', linewidth=0.5)
    ax.bar(x, soft_means, width, label='ACMR-Soft', color='#2E7D32', edgecolor='black', linewidth=0.5)
    ax.bar(x + width, static_means, width, label='Static partition', color='#DD8452', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_apps, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)

fig.suptitle('ACMR vs. ACMR-Soft vs. Static Partition, by Real Application\n(5-12 instances per application; see file list for sizes)', fontsize=12)
plt.tight_layout()
plt.savefig('fig6_per_application.png', dpi=150)
plt.close(fig)
print('saved fig6_per_application.png')


# ---- Figure 7: Pareto frontier, energy vs. violations (per-algorithm means with error bars) ----
# Raw per-instance points aren't informative here: violations are binary per
# real instance and energy spans orders of magnitude across workflow sizes,
# so a raw scatter just overplots into two flat bands. An aggregate view
# (mean +/- SD in both dimensions) is the interpretable version of this plot.
fig, ax = plt.subplots(figsize=(7.5, 6))
markers = ['o', 's', '^', 'D', 'P', 'X']
for algo, label, color, marker in zip(ALGOS, LABELS, COLORS, markers):
    e_mean, e_sd = np.mean(results[algo]['energy']), np.std(results[algo]['energy'])
    v_mean, v_sd = np.mean(results[algo]['violations']), np.std(results[algo]['violations'])
    ax.errorbar(e_mean, v_mean, xerr=e_sd, yerr=v_sd, fmt=marker, color=color,
                markersize=14, capsize=5, elinewidth=1.5, markeredgecolor='black',
                markeredgewidth=1.2, label=label.replace('\n', ' '), alpha=0.85)

ax.set_xlabel('Mean energy consumption (model units)')
ax.set_ylabel('Mean deadline violations (fraction of instances)')
ax.set_title('Energy-vs-Violations Trade-off\n(60 real instances; error bars = \u00b11 SD)')
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('fig7_pareto_scatter.png', dpi=150)
plt.close(fig)
print('saved fig7_pareto_scatter.png')

print('\nAll 7 figures generated.')
