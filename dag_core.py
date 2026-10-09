"""
DAG-generalized scheduling core, for running the ACMR comparison suite on
REAL workflow execution traces (Montage, Epigenomics, 1000Genome, SoyKB,
Seismology -- from WfCommons/WfInstances, real measured runtimes from
Chameleon Cloud executions) instead of synthetic linear-chain tasks.

WHY THIS FILE EXISTS: the paper's own formal model (Eq. 11) only allows a
task to be a strict linear chain of modules -- "finish(M_k,l) <= start(M_k,l+1)"
for ALL l, i.e. a total order. Real scientific workflows are branching DAGs
(a module can have multiple parents/children). This is a genuine, necessary
generalization, not an optional add-on: real workflow benchmarks simply
cannot be represented in the paper's stated model otherwise. This module
extends -- not replaces -- acmr_core.py:
  - Tables I/II, Eq. 4-10 (processors, DVFS modes, energy/reliability model,
    load-pressure score) are reused UNCHANGED from acmr_core.
  - Eq. 11 (precedence) is generalized from "module i's only predecessor is
    module i-1" to "module i's ready time is the max finish time over ALL
    of its parents" -- a strict superset; a linear chain is the special case
    where every module has exactly one parent (the previous module).
  - Eq. 16 (HEFT upward rank) is generalized from the chain recursion to the
    standard DAG upward-rank recursion: rank(n) = avg_exec(n) +
    max_{child in children(n)} rank(child), rank=avg_exec for exit nodes.
  - Eq. 19 (deadline) is generalized from "3.5x total serial chain length" to
    "3.5x critical-path length" (the DAG's longest path by average execution
    time) -- identical formula, applied to the DAG's actual bottleneck path
    instead of assuming the whole task is one path.
"""

import json
import numpy as np
from collections import deque

from acmr_core import (SPEED, BASEPWR, LAMBDA_BASE, N_PROC,
                        F_MULT, V_MULT, PHI_MULT, N_MODES,
                        ALPHA, BETA_W, GAMMA, LAMBDA_PEN, R_MIN, ENERGY_NORM,
                        SWARM_N, ITERS, BETA_START, BETA_END,
                        THETA_H, THETA_L, EPS_CLIP,
                        load_pressure, manifold_width, mode_bias,
                        most_congested_processor)

# Real measured runtimes are in wall-clock seconds on a specific real
# machine (Chameleon Cloud). The paper's model uses abstract "cycles"
# (w_i), executed at s_j * f_m cycles/second (Eq. 7). To place real
# runtimes into that unit system we need ONE documented conversion
# assumption: we treat the measured runtime as having been produced at
# the model's mean processor speed running in mode S2 (the same reference
# point the paper's own synthetic deadline formula, Eq. 19, already uses).
# This is an explicit, flagged assumption -- there is no way to avoid one,
# since the real traces were not run on our Table I processors.
REF_CYCLES_PER_SEC = SPEED.mean() * F_MULT[1]


# ============================================================================
# REAL WORKFLOW LOADER
# ============================================================================
class DAGTask:
    """One workflow instance: a DAG of modules with precedence + workload."""
    __slots__ = ('name', 'ids', 'idx', 'parents', 'children', 'cycles',
                 'topo_order', 'exit_idx', 'critical_path_cycles')

    def __init__(self, name, ids, parents, children, cycles):
        self.name = name
        self.ids = ids                        # list of node ids, index = local module index
        self.idx = {nid: i for i, nid in enumerate(ids)}
        self.parents = parents                # dict local_idx -> list[local_idx]
        self.children = children              # dict local_idx -> list[local_idx]
        self.cycles = cycles                  # np.array, local_idx -> workload (abstract cycles)
        self.topo_order = _topological_order(len(ids), parents, children)
        self.exit_idx = [i for i in range(len(ids)) if not children[i]]
        self.critical_path_cycles = _critical_path_length(
            len(ids), self.topo_order, children, cycles)

    def __len__(self):
        return len(self.ids)


def _topological_order(n, parents, children):
    indeg = [len(parents[i]) for i in range(n)]
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    indeg = indeg[:]
    while q:
        u = q.popleft()
        order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != n:
        raise ValueError("Workflow graph has a cycle or is disconnected unexpectedly")
    return order


def _critical_path_length(n, topo_order, children, cycles):
    """Longest path by summed workload (cycles), used for the deadline formula."""
    longest = np.zeros(n)
    for u in reversed(topo_order):
        if not children[u]:
            longest[u] = cycles[u]
        else:
            longest[u] = cycles[u] + max(longest[c] for c in children[u])
    return float(max(longest)) if n > 0 else 0.0


def load_real_workflow(json_path):
    """Parse one WfCommons/WfInstances JSON trace into a DAGTask."""
    with open(json_path) as f:
        d = json.load(f)
    spec_tasks = d['workflow']['specification']['tasks']
    exec_tasks = {t['id']: t for t in d['workflow']['execution']['tasks']}

    ids = [t['id'] for t in spec_tasks]
    idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)

    parents = {i: [] for i in range(n)}
    children = {i: [] for i in range(n)}
    for t in spec_tasks:
        i = idx[t['id']]
        for c in t.get('children', []):
            if c in idx:  # guard against dangling references
                children[i].append(idx[c])
        for p in t.get('parents', []):
            if p in idx:
                parents[i].append(idx[p])

    cycles = np.zeros(n)
    for t in spec_tasks:
        i = idx[t['id']]
        rt = exec_tasks.get(t['id'], {}).get('runtimeInSeconds', None)
        if rt is None or rt <= 0:
            rt = 0.01  # guard against zero/missing runtime entries
        cycles[i] = rt * REF_CYCLES_PER_SEC

    name = d.get('name', json_path.split('/')[-1])
    return DAGTask(name, ids, parents, children, cycles)


def deadline_for(dag_task, slack=3.5):
    """Generalizes Eq. 19 to the DAG's actual critical path."""
    s_bar = SPEED.mean()
    f2 = F_MULT[1]
    return slack * dag_task.critical_path_cycles / (s_bar * f2)


# ============================================================================
# WORKLOAD BUILDER (K real workflow instances submitted concurrently)
# ============================================================================
def build_real_workload(json_paths, rng, k_tasks):
    """Sample k_tasks real workflow files (with replacement if needed) to form
    one trial's concurrent batch, mirroring generate_workload()'s role for
    synthetic data. Returns (tasks, deadlines) in the same shape convention
    as acmr_core, except each "task" is now a DAGTask, not a flat cycle array."""
    chosen = rng.choice(json_paths, size=k_tasks, replace=(k_tasks > len(json_paths)))
    tasks = [load_real_workflow(p) for p in chosen]
    deadlines = np.array([deadline_for(t) for t in tasks])
    return tasks, deadlines


# ============================================================================
# SCHEDULE EVALUATION (DAG-generalized EFTB, Eq. 7-10)
# ============================================================================
def build_schedule_dag(tasks, assign_proc, assign_mode):
    """
    assign_proc/assign_mode: list per task of arrays (len = len(dag_task))
    giving processor/mode index per LOCAL module index.
    Modules execute in a fixed topological order per task (precedence is
    structural, independent of assignment); a module's ready time is the
    max finish time over ALL its parents (Eq. 11, generalized).
    """
    proc_free = np.zeros(N_PROC)
    proc_busy = np.zeros(N_PROC)
    finish_times = np.zeros(len(tasks))
    energy_total = 0.0
    reliab_per_module = []

    for ti, dag in enumerate(tasks):
        finish = np.zeros(len(dag))
        for local_i in dag.topo_order:
            p = assign_proc[ti][local_i]
            m = assign_mode[ti][local_i]
            w_i = dag.cycles[local_i]

            t_ijm = w_i / (SPEED[p] * F_MULT[m])
            e_ijm = BASEPWR[p] * (V_MULT[m] ** 2) * F_MULT[m] * t_ijm
            r_ijm = np.exp(-LAMBDA_BASE[p] * PHI_MULT[m] * t_ijm)

            ready = 0.0
            for par in dag.parents[local_i]:
                ready = max(ready, finish[par])

            start = max(ready, proc_free[p])
            end = start + t_ijm
            proc_free[p] = end
            proc_busy[p] += t_ijm
            finish[local_i] = end

            energy_total += e_ijm
            reliab_per_module.append(r_ijm)

        finish_times[ti] = max(finish[e] for e in dag.exit_idx) if dag.exit_idx else 0.0

    r_sys = float(np.prod(reliab_per_module))
    makespan = float(proc_free.max()) if proc_free.max() > 0 else 1e-9
    return finish_times, energy_total, r_sys, proc_busy, makespan


def evaluate_dag(tasks, deadlines, assign_proc, assign_mode):
    finish_times, energy, r_sys, proc_busy, makespan = build_schedule_dag(
        tasks, assign_proc, assign_mode)

    util = proc_busy / makespan
    limb = float(np.sqrt(np.mean((util - util.mean()) ** 2)))

    overrun = np.maximum(0.0, finish_times - deadlines) / deadlines
    p_dl = float(overrun.sum())
    violations = int(np.sum(finish_times > deadlines))

    reliab_penalty = 50.0 * max(0.0, R_MIN - r_sys)
    f_obj = (ALPHA * (energy / ENERGY_NORM) + BETA_W * (1 - r_sys) + GAMMA * limb
              + LAMBDA_PEN * p_dl + reliab_penalty)

    return dict(energy=energy, reliability=r_sys, load_imbalance=limb,
                violations=violations, f_obj=f_obj, proc_busy=proc_busy,
                makespan=makespan, finish_times=finish_times,
                assign_proc=assign_proc, assign_mode=assign_mode)


# ============================================================================
# HEFT (DAG-generalized upward rank, Eq. 16)
# ============================================================================
def heft_schedule_dag(tasks, deadlines):
    mode_s2 = 1
    avg_speed = SPEED.mean() * F_MULT[mode_s2]

    all_ranks = []
    for dag in tasks:
        avg_exec = dag.cycles / avg_speed
        rank = np.zeros(len(dag))
        for local_i in reversed(dag.topo_order):
            if not dag.children[local_i]:
                rank[local_i] = avg_exec[local_i]
            else:
                rank[local_i] = avg_exec[local_i] + max(rank[c] for c in dag.children[local_i])
        all_ranks.append(rank)

    proc_free = np.zeros(N_PROC)
    proc_busy = np.zeros(N_PROC)
    finish = [np.zeros(len(dag)) for dag in tasks]
    assign_proc = [[-1] * len(dag) for dag in tasks]
    assign_mode = [[mode_s2] * len(dag) for dag in tasks]
    energy_total = 0.0
    reliab_per_module = []

    # global candidate set: for each task, modules whose parents are ALL
    # already scheduled become "ready"; pick the ready module with the
    # highest rank across all tasks, each iteration.
    scheduled = [np.zeros(len(dag), dtype=bool) for dag in tasks]
    n_left = sum(len(dag) for dag in tasks)

    def ready_modules():
        for ti, dag in enumerate(tasks):
            for local_i in range(len(dag)):
                if not scheduled[ti][local_i] and all(scheduled[ti][p] for p in dag.parents[local_i]):
                    yield ti, local_i

    while n_left > 0:
        best = None
        best_rank = -np.inf
        for ti, local_i in ready_modules():
            if all_ranks[ti][local_i] > best_rank:
                best_rank = all_ranks[ti][local_i]
                best = (ti, local_i)
        ti, local_i = best
        dag = tasks[ti]
        w_i = dag.cycles[local_i]

        ready = 0.0
        for par in dag.parents[local_i]:
            ready = max(ready, finish[ti][par])

        best_p, best_end, best_exec = None, None, None
        for p in range(N_PROC):
            t_ijm = w_i / (SPEED[p] * F_MULT[mode_s2])
            start = max(ready, proc_free[p])
            end = start + t_ijm
            if best_end is None or end < best_end:
                best_p, best_end, best_exec = p, end, t_ijm

        assign_proc[ti][local_i] = best_p
        proc_free[best_p] = best_end
        proc_busy[best_p] += best_exec
        finish[ti][local_i] = best_end
        scheduled[ti][local_i] = True
        n_left -= 1

        e_ijm = BASEPWR[best_p] * (V_MULT[mode_s2] ** 2) * F_MULT[mode_s2] * best_exec
        r_ijm = np.exp(-LAMBDA_BASE[best_p] * PHI_MULT[mode_s2] * best_exec)
        energy_total += e_ijm
        reliab_per_module.append(r_ijm)

    finish_times = np.array([max(finish[ti][e] for e in tasks[ti].exit_idx)
                              for ti in range(len(tasks))])
    r_sys = float(np.prod(reliab_per_module))
    makespan = float(proc_free.max())
    util = proc_busy / makespan
    limb = float(np.sqrt(np.mean((util - util.mean()) ** 2)))
    overrun = np.maximum(0.0, finish_times - deadlines) / deadlines
    p_dl = float(overrun.sum())
    violations = int(np.sum(finish_times > deadlines))
    reliab_penalty = 50.0 * max(0.0, R_MIN - r_sys)
    f_obj = (ALPHA * (energy_total / ENERGY_NORM) + BETA_W * (1 - r_sys) + GAMMA * limb
              + LAMBDA_PEN * p_dl + reliab_penalty)

    return dict(energy=energy_total, reliability=r_sys, load_imbalance=limb,
                violations=violations, f_obj=f_obj, proc_busy=proc_busy,
                makespan=makespan, finish_times=finish_times,
                assign_proc=assign_proc, assign_mode=assign_mode)


# ============================================================================
# QPSO ENCODE/DECODE (same scheme as acmr_core, generalized to DAG module count)
# ============================================================================
def decode_dag(tasks, positions, manifold):
    assign_proc, assign_mode = [], []
    for ti, dag in enumerate(tasks):
        n = len(dag)
        procs = np.minimum((positions[ti][:, 0] * N_PROC).astype(int), N_PROC - 1)
        modes = np.zeros(n, dtype=int)
        for local_i in range(n):
            allowed = manifold[(ti, local_i)] if manifold is not None else [0, 1, 2]
            j = min(int(positions[ti][local_i, 1] * len(allowed)), len(allowed) - 1)
            modes[local_i] = allowed[j]
        assign_proc.append(procs)
        assign_mode.append(modes)
    return assign_proc, assign_mode


def heft_seed_particle_dag(tasks, assign_proc, assign_mode):
    particle = []
    for ti, dag in enumerate(tasks):
        n = len(dag)
        arr = np.zeros((n, 2))
        for local_i in range(n):
            arr[local_i, 0] = (assign_proc[ti][local_i] + 0.5) / N_PROC
            arr[local_i, 1] = (assign_mode[ti][local_i] + 0.5) / N_MODES
        particle.append(arr)
    return particle


def run_qpso_dag(tasks, deadlines, rng, variant="baseline", seed_particle=None,
                  reshape_start_iter=0, bias_strength=0.10):
    """Same algorithm as acmr_core.run_qpso; decode/build_schedule swapped
    for the DAG-generalized versions. See acmr_core.run_qpso for the
    per-variant documentation (baseline/static/acmr/acmr_soft/etc.)."""
    dims = [(len(dag), 2) for dag in tasks]

    def full_manifold():
        return {(ti, local_i): [0, 1, 2] for ti, dag in enumerate(tasks) for local_i in range(len(dag))}

    def random_manifold(rng_):
        options = [[0, 1, 2], [0, 1], [1, 2]]
        return {(ti, local_i): options[rng_.integers(3)]
                for ti, dag in enumerate(tasks) for local_i in range(len(dag))}

    particles = [[rng.random(d) for d in dims] for _ in range(SWARM_N)]
    if seed_particle is not None:
        particles[0] = [p.copy() for p in seed_particle]
    manifold = full_manifold() if variant not in ("baseline", "acmr_soft") else None
    bias = None

    pbest = [[p[k].copy() for k in range(len(p))] for p in particles]
    pbest_val = [None] * SWARM_N
    gbest, gbest_val, gbest_assign = None, None, None

    for t in range(ITERS):
        beta = BETA_START + (BETA_END - BETA_START) * t / (ITERS - 1)

        if t > 0 and variant != "baseline":
            ap, am = decode_dag(tasks, gbest, manifold)
            _, _, _, proc_busy, makespan = build_schedule_dag(tasks, ap, am)
            if variant == "acmr":
                if t >= reshape_start_iter:
                    manifold = manifold_width(proc_busy, makespan, ap, tasks)
            elif variant == "static":
                pass
            elif variant == "ablation_random":
                manifold = random_manifold(rng)
            elif variant == "acmr_soft":
                bias = mode_bias(proc_busy, makespan, ap, tasks)
            elif variant == "acmr_targeted":
                worst_p, lp = most_congested_processor(proc_busy, makespan)
                new_manifold = full_manifold()
                if lp[worst_p] >= THETA_H:
                    for ti, dag in enumerate(tasks):
                        for local_i in range(len(dag)):
                            if ap[ti][local_i] == worst_p:
                                new_manifold[(ti, local_i)] = [0, 1]
                manifold = new_manifold

        if t == 0 and variant == "static":
            best_i, best_v = None, None
            for i in range(SWARM_N):
                ap0, am0 = decode_dag(tasks, particles[i], full_manifold())
                r0 = evaluate_dag(tasks, deadlines, ap0, am0)
                if best_v is None or r0['f_obj'] < best_v:
                    best_v, best_i = r0['f_obj'], i
            ap0, am0 = decode_dag(tasks, particles[best_i], full_manifold())
            _, _, _, proc_busy0, makespan0 = build_schedule_dag(tasks, ap0, am0)
            manifold = manifold_width(proc_busy0, makespan0, ap0, tasks)

        for i in range(SWARM_N):
            ap, am = decode_dag(tasks, particles[i], manifold)
            res = evaluate_dag(tasks, deadlines, ap, am)
            if pbest_val[i] is None or res['f_obj'] < pbest_val[i]:
                pbest_val[i] = res['f_obj']
                pbest[i] = [particles[i][k].copy() for k in range(len(particles[i]))]
            if gbest_val is None or res['f_obj'] < gbest_val:
                gbest_val = res['f_obj']
                gbest = [particles[i][k].copy() for k in range(len(particles[i]))]
                gbest_assign = (ap, am)

        mbest = [np.mean([pbest[i][k] for i in range(SWARM_N)], axis=0) for k in range(len(dims))]
        for i in range(SWARM_N):
            for k in range(len(dims)):
                phi = rng.random(dims[k])
                attractor = phi * pbest[i][k] + (1 - phi) * gbest[k]
                u = rng.random(dims[k])
                sign = rng.choice([-1, 1], size=dims[k])
                particles[i][k] = np.clip(
                    attractor + sign * beta * np.abs(mbest[k] - particles[i][k]) * np.log(1.0 / (u + 1e-12)),
                    0, 1 - EPS_CLIP)

            if variant == "acmr_soft" and bias is not None:
                for ti, dag in enumerate(tasks):
                    for local_i in range(len(dag)):
                        b = bias.get((ti, local_i), 0.0)
                        if b != 0.0:
                            particles[i][ti][local_i, 1] = np.clip(
                                particles[i][ti][local_i, 1] - b * bias_strength * beta,
                                0, 1 - EPS_CLIP)

    ap, am = decode_dag(tasks, gbest, manifold)
    return evaluate_dag(tasks, deadlines, ap, am)
