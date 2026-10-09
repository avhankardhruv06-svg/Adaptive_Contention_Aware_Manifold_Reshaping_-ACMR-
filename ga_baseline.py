"""
Genetic Algorithm baseline for the ACMR comparison suite.

Uses the SAME real-valued encoding (Eq. 17-18), the SAME objective (Eq. 12),
and the SAME evaluation budget (population 30 x 60 generations = 1800
evaluations, identical to QPSO's 30 particles x 60 iterations) as the QPSO
variants, so the comparison is budget-matched rather than favoring either
side by evaluation count.

Standard GA operators: tournament selection, uniform crossover,
Gaussian-perturbation mutation, elitism of 1. No paper-specific tuning
beyond matching population/budget to the other algorithms.
"""

import numpy as np
from acmr_core import decode, evaluate, SWARM_N, ITERS


def _flatten(particle):
    return np.concatenate([p.ravel() for p in particle])


def _unflatten(flat, dims):
    out, idx = [], 0
    for d in dims:
        n = d[0] * d[1]
        out.append(flat[idx:idx + n].reshape(d))
        idx += n
    return out


def run_ga(tasks, deadlines, rng, pop_size=SWARM_N, generations=ITERS,
           tournament_k=3, crossover_p=0.9, mutation_p=0.05, mutation_sigma=0.15,
           elitism=1):
    dims = [(len(c), 2) for c in tasks]
    genome_len = sum(d[0] * d[1] for d in dims)

    pop = [rng.random(genome_len) for _ in range(pop_size)]

    def fitness(genome):
        particle = _unflatten(genome, dims)
        ap, am = decode(tasks, particle, None)  # unrestricted manifold, same as baseline QPSO
        return evaluate(tasks, deadlines, ap, am)['f_obj']

    fit = np.array([fitness(g) for g in pop])

    for gen in range(generations):
        order = np.argsort(fit)
        new_pop = [pop[i].copy() for i in order[:elitism]]  # elitism

        while len(new_pop) < pop_size:
            # tournament selection (x2 parents)
            def tournament():
                idxs = rng.integers(0, pop_size, size=tournament_k)
                best = idxs[np.argmin(fit[idxs])]
                return pop[best]

            p1, p2 = tournament(), tournament()

            if rng.random() < crossover_p:
                mask = rng.random(genome_len) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()

            mut_mask = rng.random(genome_len) < mutation_p
            noise = rng.normal(0, mutation_sigma, size=genome_len)
            child = np.clip(child + mut_mask * noise, 0, 1 - 1e-6)

            new_pop.append(child)

        pop = new_pop[:pop_size]
        fit = np.array([fitness(g) for g in pop])

    best_idx = np.argmin(fit)
    best_particle = _unflatten(pop[best_idx], dims)
    ap, am = decode(tasks, best_particle, None)
    return evaluate(tasks, deadlines, ap, am)
