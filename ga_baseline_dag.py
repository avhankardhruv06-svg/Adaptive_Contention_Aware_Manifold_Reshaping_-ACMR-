"""GA baseline for the DAG-generalized (real workflow) comparison suite."""

import numpy as np
from dag_core import decode_dag, evaluate_dag
from acmr_core import SWARM_N, ITERS


def _flatten(particle):
    return np.concatenate([p.ravel() for p in particle])


def _unflatten(flat, dims):
    out, idx = [], 0
    for d in dims:
        n = d[0] * d[1]
        out.append(flat[idx:idx + n].reshape(d))
        idx += n
    return out


def run_ga_dag(tasks, deadlines, rng, pop_size=SWARM_N, generations=ITERS,
               tournament_k=3, crossover_p=0.9, mutation_p=0.05, mutation_sigma=0.15,
               elitism=1):
    dims = [(len(dag), 2) for dag in tasks]
    genome_len = sum(d[0] * d[1] for d in dims)

    pop = [rng.random(genome_len) for _ in range(pop_size)]

    def fitness(genome):
        particle = _unflatten(genome, dims)
        ap, am = decode_dag(tasks, particle, None)
        return evaluate_dag(tasks, deadlines, ap, am)['f_obj']

    fit = np.array([fitness(g) for g in pop])

    for gen in range(generations):
        order = np.argsort(fit)
        new_pop = [pop[i].copy() for i in order[:elitism]]

        while len(new_pop) < pop_size:
            def tournament():
                idxs = rng.integers(0, pop_size, size=tournament_k)
                return pop[idxs[np.argmin(fit[idxs])]]

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

    best_particle = _unflatten(pop[np.argmin(fit)], dims)
    ap, am = decode_dag(tasks, best_particle, None)
    return evaluate_dag(tasks, deadlines, ap, am)
