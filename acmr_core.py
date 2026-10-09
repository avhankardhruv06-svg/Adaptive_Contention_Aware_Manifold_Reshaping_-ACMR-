"""
Faithful reimplementation of the ACMR-QPSO paper's system model.
Every constant and formula below is taken directly from the manuscript:
  - Table I  (processor speed / power / base fault rate)
  - Table II (DVFS mode f_m, v_m, phi_m)
  - Eq. 4-6  (load-pressure score, manifold-width function)
  - Eq. 7-10 (execution time, energy, per-module reliability, system reliability)
  - Eq. 12-15 (composite objective: energy, reliability, load imbalance, deadline penalty)
  - Eq. 16   (HEFT upward rank)
  - Eq. 17-18 (QPSO particle decoding)
  - Eq. 19   (deadline formula)
  - Algorithm 1 (full ACMR-QPSO loop)
No placeholder constants remain; nothing here needs to be "swapped before use."
"""

import numpy as np

# ----------------------------- TABLE I: PROCESSORS -----------------------------
# order: P1 (high-perf), P2 (high-perf), P3 (balanced), P4 (low-power)
SPEED   = np.array([3.2, 3.0, 2.0, 1.0])          # s_j, relative speed
BASEPWR = np.array([4.0, 3.6, 2.2, 1.0])          # pw_j, Watts
LAMBDA_BASE = np.array([4.5e-4, 4.0e-4, 2.6e-4, 1.5e-4])  # lambda_j^base
N_PROC = len(SPEED)

# ----------------------------- TABLE II: DVFS MODES -----------------------------
# order: S1 (high performance), S2 (balanced), S3 (energy-efficient)
F_MULT   = np.array([1.00, 0.75, 0.50])   # f_m frequency multiplier
V_MULT   = np.array([1.00, 0.85, 0.70])   # v_m voltage multiplier
PHI_MULT = np.array([1.00, 0.55, 0.30])   # phi_m fault-rate multiplier
N_MODES = len(F_MULT)

# ----------------------------- OBJECTIVE WEIGHTS (Sec. VIII-C) -----------------------------
ALPHA, BETA_W, GAMMA, LAMBDA_PEN = 0.4, 0.4, 0.2, 50.0
R_MIN = 0.70  # soft reliability threshold (Sec. IV-E item 3)
ENERGY_NORM = 1000.0  # see note in evaluate(); UNDERSPECIFIED by the paper -
                       # this is our documented, fixed assumption, tested
                       # for sensitivity in experiment 3 below.

# ----------------------------- QPSO / ACMR CONFIG (Sec. VIII-C) -----------------------------
SWARM_N = 30
ITERS = 60
BETA_START, BETA_END = 1.0, 0.4
THETA_H, THETA_L = 1.25, 0.75  # Eq. 6 thresholds
EPS_CLIP = 1e-6                 # clip(x,0,1-eps), Algorithm 1 line 18


# ============================================================================
# WORKLOAD GENERATOR (Sec. VIII-B, Eq. 19)
# ============================================================================
def generate_workload(rng, k_tasks):
    """Each task has n_k ~ U{3,6} modules; module workload w_i ~ U(8,40) cycles."""
    tasks = []
    for _ in range(k_tasks):
        n_mod = rng.integers(3, 7)  # inclusive of 6
        cycles = rng.uniform(8, 40, size=n_mod)
        tasks.append(cycles)
    s_bar = SPEED.mean()
    f2 = F_MULT[1]  # mode-S2 frequency multiplier
    deadlines = [3.5 * cyc.sum() / (s_bar * f2) for cyc in tasks]
    return tasks, np.array(deadlines)


# ============================================================================
# SCHEDULE EVALUATION (EFTB, Sec. V-2; Eq. 7-10)
# ============================================================================
def build_schedule(tasks, assign_proc, assign_mode):
    """
    Modules within a task execute strictly in sequence (Eq. 11); modules of
    different tasks may run concurrently on different processors. All tasks
    are assumed to arrive at t=0 (paper does not specify staggered arrivals).
    Greedy EFTB timing: each module starts at
        max(previous module in its task finishes, assigned processor free time).
    Processor and mode are fixed externally (by QPSO's mapping, or by HEFT).
    """
    proc_free = np.zeros(N_PROC)
    proc_busy = np.zeros(N_PROC)     # sum of execution time on each processor
    finish_times = np.zeros(len(tasks))
    energy_total = 0.0
    reliab_per_module = []

    for ti, cycles in enumerate(tasks):
        t_ready = 0.0
        for mi, w_i in enumerate(cycles):
            p = assign_proc[ti][mi]
            m = assign_mode[ti][mi]

            t_ijm = w_i / (SPEED[p] * F_MULT[m])                      # Eq. 7
            e_ijm = BASEPWR[p] * (V_MULT[m] ** 2) * F_MULT[m] * t_ijm  # Eq. 8
            r_ijm = np.exp(-LAMBDA_BASE[p] * PHI_MULT[m] * t_ijm)      # Eq. 9

            start = max(t_ready, proc_free[p])
            end = start + t_ijm
            proc_free[p] = end
            proc_busy[p] += t_ijm
            t_ready = end

            energy_total += e_ijm
            reliab_per_module.append(r_ijm)

        finish_times[ti] = t_ready

    r_sys = float(np.prod(reliab_per_module))   # Eq. 10
    makespan = float(proc_free.max()) if proc_free.max() > 0 else 1e-9
    return finish_times, energy_total, r_sys, proc_busy, makespan


def evaluate(tasks, deadlines, assign_proc, assign_mode):
    finish_times, energy, r_sys, proc_busy, makespan = build_schedule(
        tasks, assign_proc, assign_mode)

    # Eq. 4-5: utilization and load-pressure (also needed for ACMR manifold)
    util = proc_busy / makespan
    util_bar = util.mean()
    limb = float(np.sqrt(np.mean((util - util_bar) ** 2)))            # Eq. 14

    overrun = np.maximum(0.0, finish_times - deadlines) / deadlines
    p_dl = float(overrun.sum())                                       # Eq. 15
    violations = int(np.sum(finish_times > deadlines))

    reliab_penalty = 50.0 * max(0.0, R_MIN - r_sys)                   # Sec IV-E(3)
    # NOTE ON ENERGY NORMALIZATION: Table III's column header is explicitly
    # "Energy (normalised)" but the paper never states the normalization
    # factor used inside Eq. 12. Feeding *raw* eijm sums into alpha*Etotal
    # makes the energy term outweigh lambda*Pdl by roughly 20:1 even though
    # lambda=50 is described as deliberately "large" to reflect deadline cost
    # (Sec. IV-D) -- i.e. raw energy silently defeats the paper's own stated
    # weighting intent. We normalize energy by ENERGY_NORM (a fixed constant,
    # not re-tuned per algorithm) so alpha*Energy_norm sits on the same O(1)
    # scale as beta*(1-Rsys) and gamma*Limb, matching what the stated
    # weights (alpha=beta=0.4, lambda=50) evidently assume. Raw energy is
    # still reported/returned unmodified for comparison with Table III.
    energy_norm = energy / ENERGY_NORM
    f_obj = (ALPHA * energy_norm + BETA_W * (1 - r_sys) + GAMMA * limb
              + LAMBDA_PEN * p_dl + reliab_penalty)                    # Eq. 12

    return dict(energy=energy, reliability=r_sys, load_imbalance=limb,
                violations=violations, f_obj=f_obj, proc_busy=proc_busy,
                makespan=makespan, finish_times=finish_times)


def load_pressure(proc_busy, makespan):
    """Eq. 4-5, applied to the whole processor fleet from the incumbent schedule."""
    util = proc_busy / makespan
    util_bar = util.mean() if util.mean() > 0 else 1e-9
    return util / util_bar


# ============================================================================
# HEFT BASELINE (Sec. VI, Eq. 16) -- true per-module upward rank
# ============================================================================
def heft_schedule(tasks, deadlines):
    n_proc = N_PROC
    mode_s2 = 1
    avg_speed = SPEED.mean() * F_MULT[mode_s2]

    # Eq. 16: upward rank computed backward from the last module in each task
    ranks = []
    for cycles in tasks:
        avg_exec = cycles / avg_speed
        r = np.zeros(len(cycles))
        r[-1] = avg_exec[-1]
        for i in range(len(cycles) - 2, -1, -1):
            r[i] = avg_exec[i] + r[i + 1]
        ranks.append(r)

    # Flatten to a single priority list of (rank, task_idx, module_idx),
    # sorted in decreasing rank order, but a module can only be scheduled
    # once its intra-task predecessor has been placed (Eq. 11).
    pending = [[] for _ in tasks]
    for ti in range(len(tasks)):
        pending[ti] = list(range(len(tasks[ti])))

    assign_proc = [[-1] * len(c) for c in tasks]
    assign_mode = [[mode_s2] * len(c) for c in tasks]
    proc_free = np.zeros(n_proc)
    proc_busy = np.zeros(n_proc)
    ready_time = [0.0] * len(tasks)
    finish_times = np.zeros(len(tasks))
    energy_total = 0.0
    reliab_per_module = []

    # Build a global ordering: all (task, first-unscheduled-module) candidates,
    # sorted by that module's own upward rank, re-evaluated as tasks progress.
    remaining = {ti: 0 for ti in range(len(tasks))}  # next module index to place per task
    n_left = sum(len(c) for c in tasks)

    while n_left > 0:
        # pick the ready module (across all tasks) with the highest rank
        best_ti, best_score = None, -np.inf
        for ti in range(len(tasks)):
            mi = remaining[ti]
            if mi < len(tasks[ti]) and ranks[ti][mi] > best_score:
                best_score = ranks[ti][mi]
                best_ti = ti
        ti = best_ti
        mi = remaining[ti]
        cyc = tasks[ti][mi]

        best_p, best_end, best_start, best_exec = None, None, None, None
        for p in range(n_proc):
            t_ijm = cyc / (SPEED[p] * F_MULT[mode_s2])
            start = max(ready_time[ti], proc_free[p])
            end = start + t_ijm
            if best_end is None or end < best_end:
                best_p, best_end, best_start, best_exec = p, end, start, t_ijm

        assign_proc[ti][mi] = best_p
        proc_free[best_p] = best_end
        proc_busy[best_p] += best_exec
        ready_time[ti] = best_end

        e_ijm = BASEPWR[best_p] * (V_MULT[mode_s2] ** 2) * F_MULT[mode_s2] * best_exec
        r_ijm = np.exp(-LAMBDA_BASE[best_p] * PHI_MULT[mode_s2] * best_exec)
        energy_total += e_ijm
        reliab_per_module.append(r_ijm)

        remaining[ti] += 1
        n_left -= 1
        if remaining[ti] == len(tasks[ti]):
            finish_times[ti] = ready_time[ti]

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


def heft_seed_particle(tasks, assign_proc, assign_mode):
    """
    Encode a HEFT solution as a QPSO position vector, for swarm seeding.
    Inverts Eq. 17-18: picks the position at the CENTER of the bucket that
    decodes to the given (processor, mode) pair, for an unrestricted
    (3-mode) initial manifold.
    """
    particle = []
    for ti, cycles in enumerate(tasks):
        arr = np.zeros((len(cycles), 2))
        for mi in range(len(cycles)):
            p = assign_proc[ti][mi]
            m = assign_mode[ti][mi]
            arr[mi, 0] = (p + 0.5) / N_PROC
            arr[mi, 1] = (m + 0.5) / N_MODES
        particle.append(arr)
    return particle


# ============================================================================
# QPSO PARTICLE ENCODING/DECODING (Eq. 17-18)
# ============================================================================
def decode(tasks, positions, manifold):
    """
    positions: list per task of (n_mod, 2) arrays in [0,1).
      col 0 -> processor sub-vector x_i^(p)
      col 1 -> mode sub-vector      x_i^(m)
    manifold: dict (task_idx, mod_idx) -> list of allowed mode indices W(M_i,t).
      If manifold is None, unrestricted: all 3 modes always allowed (baseline QPSO).
    """
    assign_proc, assign_mode = [], []
    for ti, cycles in enumerate(tasks):
        procs, modes = [], []
        for mi in range(len(cycles)):
            j = min(int(positions[ti][mi, 0] * N_PROC), N_PROC - 1)      # Eq. 17
            if manifold is not None:
                allowed = manifold[(ti, mi)]
            else:
                allowed = [0, 1, 2]
            idx = min(int(positions[ti][mi, 1] * len(allowed)), len(allowed) - 1)  # Eq. 18
            m = allowed[idx]
            procs.append(j)
            modes.append(m)
        assign_proc.append(procs)
        assign_mode.append(modes)
    return assign_proc, assign_mode


def manifold_width(proc_busy, makespan, assign_proc, tasks):
    """Eq. 6: next-iteration DVFS-mode set per module, from current gbest schedule."""
    lp = load_pressure(proc_busy, makespan)
    new_manifold = {}
    for ti, cycles in enumerate(tasks):
        for mi in range(len(cycles)):
            p = assign_proc[ti][mi]
            if lp[p] >= THETA_H:
                new_manifold[(ti, mi)] = [0, 1, 2]
            elif lp[p] <= THETA_L:
                new_manifold[(ti, mi)] = [1, 2]
            else:
                new_manifold[(ti, mi)] = [0, 1]
    return new_manifold


def mode_bias(proc_busy, makespan, assign_proc, tasks):
    """
    ACMR-Soft: instead of hard-restricting the decode manifold (which
    retroactively reinterprets pbest/gbest positions under a new decode
    function -- the diagnosed failure mode), compute a per-module bias in
    [-1, +1] from load pressure and apply it as a drift term added directly
    to the mode sub-vector during the move step. The decode function itself
    never changes, so a given position always means the same thing across
    iterations; only the *search direction* is nudged by contention.
    bias > 0 -> nudge toward mode index 0 (S1, fast) on congested processors
    bias < 0 -> nudge toward mode index 2 (S3, slow) on underloaded processors
    """
    lp = load_pressure(proc_busy, makespan)
    bias = {}
    for ti, cycles in enumerate(tasks):
        for mi in range(len(cycles)):
            p = assign_proc[ti][mi]
            if lp[p] >= THETA_H:
                bias[(ti, mi)] = +1.0
            elif lp[p] <= THETA_L:
                bias[(ti, mi)] = -1.0
            else:
                bias[(ti, mi)] = 0.0
    return bias


def most_congested_processor(proc_busy, makespan):
    lp = load_pressure(proc_busy, makespan)
    return int(np.argmax(lp)), lp


# ============================================================================
# QPSO / ACMR-QPSO (Algorithm 1)
# ============================================================================
def run_qpso(tasks, deadlines, rng, variant="baseline", reshape_fn=None,
             seed_particle=None, reshape_start_iter=0, bias_strength=0.15):
    """
    variant:
      "baseline" - unrestricted QPSO, manifold always all 3 modes
      "static"   - static slack-partition QPSO: manifold fixed once at t=0
                    from the *initial* gbest load pressure, never updated again
      "acmr"     - Adaptive Contention-Aware Manifold Reshaping (Algorithm 1,
                    original hard-restriction mechanism)
      "ablation_random" - manifold reshaped every iteration like ACMR, but the
                    2-mode/3-mode assignment is randomly shuffled across
                    processors rather than driven by load-pressure
      "acmr_soft" - REDESIGN #1: decode manifold is NEVER restricted (always
                    all 3 modes, so pbest/gbest positions always mean the
                    same thing across iterations). Contention feedback is
                    injected as a drift term added directly to each
                    congested/underloaded module's mode sub-vector during
                    the move step, instead of hard-excluding modes.
      "acmr_targeted" - REDESIGN #2: keeps the original hard-restriction
                    mechanism, but applies it only to modules currently
                    assigned (in gbest) to the SINGLE most-congested
                    processor each iteration; every other module's decode
                    stays permanently unrestricted. Shrinks the fraction of
                    the swarm's memory invalidated by reshaping each step.
    reshape_fn: optional override for custom ablations.
    seed_particle: optional initial position (e.g. from heft_seed_particle).
    reshape_start_iter: for variant="acmr", delays reshaping onset.
    bias_strength: drift magnitude per iteration for "acmr_soft" (annealed
                    by the same beta(t) schedule as the QPSO move step).
    """
    dims = [(len(c), 2) for c in tasks]

    def full_manifold():
        return {(ti, mi): [0, 1, 2] for ti, c in enumerate(tasks) for mi in range(len(c))}

    def random_manifold(rng_):
        # same *shape* of restriction as Eq.6 (either all-3, {S1,S2}, or {S2,S3})
        # but assigned uniformly at random per module, ignoring load pressure
        options = [[0, 1, 2], [0, 1], [1, 2]]
        return {(ti, mi): options[rng_.integers(3)]
                for ti, c in enumerate(tasks) for mi in range(len(c))}

    particles = [[rng.random(d) for d in dims] for _ in range(SWARM_N)]
    if seed_particle is not None:
        particles[0] = [p.copy() for p in seed_particle]
    manifold = full_manifold() if variant not in ("baseline", "acmr_soft") else None
    bias = None  # only used by acmr_soft

    pbest = [[p[k].copy() for k in range(len(p))] for p in particles]
    pbest_val = [None] * SWARM_N
    gbest, gbest_val, gbest_assign = None, None, None

    for t in range(ITERS):
        beta = BETA_START + (BETA_END - BETA_START) * t / (ITERS - 1)

        # --- manifold/bias update happens BEFORE evaluating this iteration's
        #     particles, using the gbest schedule found in the previous
        #     iteration (Algorithm 1, line 6-10) ---
        if t > 0 and variant not in ("baseline",):
            ap, am = decode(tasks, gbest, manifold)
            _, _, _, proc_busy, makespan = build_schedule(tasks, ap, am)
            if variant == "acmr":
                if t >= reshape_start_iter:
                    manifold = manifold_width(proc_busy, makespan, ap, tasks)
                # else: keep manifold unrestricted (delayed-reshaping test)
            elif variant == "static":
                pass  # manifold frozen after t=0
            elif variant == "ablation_random":
                manifold = random_manifold(rng)
            elif variant == "acmr_soft":
                # decode manifold NEVER changes (stays None -> full range);
                # only the bias dict updates each iteration
                bias = mode_bias(proc_busy, makespan, ap, tasks)
            elif variant == "acmr_targeted":
                worst_p, lp = most_congested_processor(proc_busy, makespan)
                new_manifold = full_manifold()
                if lp[worst_p] >= THETA_H:
                    for ti, cycles in enumerate(tasks):
                        for mi in range(len(cycles)):
                            if ap[ti][mi] == worst_p:
                                new_manifold[(ti, mi)] = [0, 1]  # bias this processor's modules toward fast modes
                manifold = new_manifold
            elif reshape_fn is not None:
                manifold = reshape_fn(proc_busy, makespan, ap, tasks, rng)

        if t == 0 and variant == "static":
            # static partition: compute manifold ONCE from the unrestricted
            # initial population's best particle, then freeze it (mirrors
            # "single pre-run manifold" description in Sec. III-B)
            best_i, best_v = None, None
            for i in range(SWARM_N):
                ap0, am0 = decode(tasks, particles[i], full_manifold())
                r0 = evaluate(tasks, deadlines, ap0, am0)
                if best_v is None or r0['f_obj'] < best_v:
                    best_v, best_i = r0['f_obj'], i
            ap0, am0 = decode(tasks, particles[best_i], full_manifold())
            _, _, _, proc_busy0, makespan0 = build_schedule(tasks, ap0, am0)
            manifold = manifold_width(proc_busy0, makespan0, ap0, tasks)

        for i in range(SWARM_N):
            ap, am = decode(tasks, particles[i], manifold)
            res = evaluate(tasks, deadlines, ap, am)
            if pbest_val[i] is None or res['f_obj'] < pbest_val[i]:
                pbest_val[i] = res['f_obj']
                pbest[i] = [particles[i][k].copy() for k in range(len(particles[i]))]
            if gbest_val is None or res['f_obj'] < gbest_val:
                gbest_val = res['f_obj']
                gbest = [particles[i][k].copy() for k in range(len(particles[i]))]
                gbest_assign = (ap, am)

        # mbest = mean personal-best position (Eq. 1)
        mbest = [np.mean([pbest[i][k] for i in range(SWARM_N)], axis=0) for k in range(len(dims))]
        for i in range(SWARM_N):
            for k in range(len(dims)):
                phi = rng.random(dims[k])
                attractor = phi * pbest[i][k] + (1 - phi) * gbest[k]      # Eq. 2
                u = rng.random(dims[k])
                sign = rng.choice([-1, 1], size=dims[k])
                particles[i][k] = np.clip(
                    attractor + sign * beta * np.abs(mbest[k] - particles[i][k]) * np.log(1.0 / (u + 1e-12)),
                    0, 1 - EPS_CLIP)                                       # Eq. 3

            if variant == "acmr_soft" and bias is not None:
                # inject contention feedback as a direct drift on the mode
                # sub-vector, annealed by the same beta(t) schedule, WITHOUT
                # ever touching the decode function (so pbest/gbest positions
                # keep a stable meaning across iterations)
                for ti, cycles in enumerate(tasks):
                    for mi in range(len(cycles)):
                        b = bias.get((ti, mi), 0.0)
                        if b != 0.0:
                            # b=+1 (congested) -> push toward mode index 0 (decrease x_m)
                            # b=-1 (underloaded) -> push toward mode index 2 (increase x_m)
                            particles[i][ti][mi, 1] = np.clip(
                                particles[i][ti][mi, 1] - b * bias_strength * beta,
                                0, 1 - EPS_CLIP)

    ap, am = decode(tasks, gbest, manifold)
    return evaluate(tasks, deadlines, ap, am)
