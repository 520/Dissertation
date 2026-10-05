"""Minimise uniform [-8,8] MAE for six negative and four positive SiLU knots.

Knot values remain exact SiLU values, with zero/identity tails. Independent
seeded searches use Gauss quadrature; dense-grid polishing and the existing
continuous maximum-error check verify the saved candidate.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit

from search_adaptive_silu_lut import (
    Limits, knots_from_gaps, grid_metrics, verify_candidate,
)

DIRECTORY = Path(__file__).resolve().parent
LIMITS = Limits(0.00642035, 0.00999424, 0.0334644)
OLD = np.array([-6.85632073, -5.02235249, -3.68964006, -1.51148640,
                -0.93682254, -0.45217847, 0., 0.61536532, 1.32721033,
                4.07592417, 6.00384166])


def main():
    nodes, weights = np.polynomial.legendre.leggauss(128)
    fractions = (nodes + 1) / 2
    best = None
    runs = []
    for seed in (42, 43, 44):
        def objective(gaps):
            k = knots_from_gaps(gaps, 6)
            if k[0] < -8 or k[-1] > 8:
                return 1 + max(-8-k[0], k[-1]-8, 0)
            edges = np.r_[-8., k, 8.]
            width = np.diff(edges)
            x = edges[:-1, None] + width[:, None] * fractions
            y = k * expit(k)
            prediction = np.empty_like(x)
            prediction[0] = 0
            prediction[-1] = x[-1]
            prediction[1:-1] = y[:-1, None] + np.diff(y)[:, None] * fractions
            error = np.abs(prediction - x * expit(x))
            mae = np.sum(width * (error @ weights)) / 32
            rmse = np.sqrt(np.sum(width * (error**2 @ weights)) / 32)
            maximum = max(error.max(), abs(y[0]), abs(k[-1]-y[-1]))
            penalty = max(rmse / LIMITS.rmse - 1, 0)**2
            penalty += max(maximum / LIMITS.max_error - 1, 0)**2
            return mae + penalty

        fit = differential_evolution(objective, [(0.08, 4.)]*10,
            seed=seed, maxiter=900, popsize=18, tol=1e-9, polish=True)
        # Dense polishing removes small quadrature errors near sign changes.
        x = np.linspace(-8, 8, 100001)
        reference = x * expit(x)
        def dense_objective(gaps):
            k = knots_from_gaps(gaps, 6)
            mae, rmse, maximum, _ = grid_metrics(k, x, reference)
            return mae + max(rmse/LIMITS.rmse-1, 0)**2 + max(maximum/LIMITS.max_error-1, 0)**2
        polished = minimize(dense_objective, fit.x, method='Powell',
            bounds=[(0.08, 4.)]*10, options={'maxiter':100, 'xtol':1e-8, 'ftol':1e-10})
        if dense_objective(polished.x) < dense_objective(fit.x):
            fit = polished
        vx = np.linspace(-8, 8, 1000001)
        candidate = verify_candidate(11, knots_from_gaps(fit.x, 6), fit, seed,
                                     6, vx, vx*expit(vx), LIMITS)
        runs.append(asdict(candidate))
        print(f'seed={seed}: MAE={candidate.mae:.10f}, RMSE={candidate.rmse:.10f}, max={candidate.max_error:.10f}, passes={candidate.passes}', flush=True)
        if candidate.passes and (best is None or candidate.mae < best.mae):
            best = candidate
    if best is None:
        raise RuntimeError('No candidate passed the error limits')
    old_metrics = grid_metrics(OLD, vx, vx*expit(vx))
    result = dict(description='MAE-optimised six negative / four positive exact-value SiLU interpolation knots plus zero',
        qualification='Seeded numerical optimum, not a proof of global optimality',
        metric_distribution='uniform grid', evaluation_range=[-8., 8.],
        limits=asdict(LIMITS), verification_samples=len(vx),
        search=dict(seeds=[42,43,44], maxiter=900, popsize=18,
                    minimum_gap=.08, maximum_gap=4., quadrature_nodes=128),
        previous_6_4=dict(knots=OLD.tolist(), mae=old_metrics[0], rmse=old_metrics[1], max_error=old_metrics[2]),
        selected=asdict(best), candidates=runs)
    destination = DIRECTORY/'adaptive_silu_6_4.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(destination, flush=True)


if __name__ == '__main__':
    main()
