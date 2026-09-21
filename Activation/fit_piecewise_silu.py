"""Search a symmetric 21-knot interpolating SiLU LUT with linear tails.

Minimize sampled worst-case absolute error; numerical search, not a proof of
global optimality. Symmetry f(-x)=f(x)-x reduces storage to 11 knots.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.special import expit


def silu(x):
    return x * expit(x)


def main():
    fractions = np.linspace(0, 1, 201)

    def knots(v):
        return np.r_[0., np.cumsum(v)]

    def objective(v):
        k = knots(v)
        y = silu(k)
        x = k[:-1, None] + np.diff(k)[:, None] * fractions
        approx = y[:-1, None] + np.diff(y)[:, None] * fractions
        # Search bounds ensure T>1.278, so tail error decreases beyond T.
        return max(np.max(np.abs(approx-silu(x))), k[-1]*expit(-k[-1]))

    fit = differential_evolution(objective, [(0.15, 1.6)]*10,
                                 seed=42, maxiter=1600, popsize=18,
                                 tol=1e-8, polish=True)
    k = knots(fit.x)
    y = silu(k)
    slopes = np.diff(y)/np.diff(k)
    intercepts = y[:-1]-slopes*k[:-1]
    # Independent dense check across the entire finite interpolation domain.
    x = np.linspace(-12, 12, 1000001)
    a = np.abs(x)
    pos = np.interp(a, k, y)
    pos = np.where(a >= k[-1], a, pos)
    predicted = np.where(x < 0, pos-a, pos)
    error = np.abs(predicted-silu(x))
    oldk = np.linspace(-5, 5, 21)
    old = np.interp(x, oldk, silu(oldk))
    old = np.where(x <= -5, 0, np.where(x >= 5, x, old))
    olderr = np.abs(old-silu(x))
    # Refine extrema on small subintervals so an inflection cannot hide peaks.
    peaks = []
    for l,r,m,b in zip(k[:-1],k[1:],slopes,intercepts):
        for u,v in zip(np.linspace(l,r,21)[:-1],np.linspace(l,r,21)[1:]):
            opt = minimize_scalar(lambda z: -abs(m*z+b-silu(z)),
                                  bounds=(u,v), method='bounded')
            peaks.append(-opt.fun)
    max_error = max(*peaks, k[-1]*expit(-k[-1]))
    result = dict(description='21 symmetric nonuniform knots, exact SiLU knot values, 20 interpolating segments, tails 0/x',
                  objective='minimize worst absolute error over real inputs',
                  qualification='numerical candidate, not proven global optimum',
                  optimizer_success=bool(fit.success), optimizer_message=str(fit.message),
                  positive_knots=k.tolist(), positive_values=y.tolist(),
                  slopes=slopes.tolist(), intercepts=intercepts.tolist(),
                  threshold=float(k[-1]), verified_max_error=float(max_error),
                  dense_max_error=float(error.max()),
                  mae_minus12_12=float(error.mean()),
                  baseline_max_error=float(olderr.max()),
                  baseline_mae_minus12_12=float(olderr.mean()),
                  positive_equations=[dict(left=float(l),right=float(r),a=float(m),b=float(b))
                                      for l,r,m,b in zip(k[:-1],k[1:],slopes,intercepts)])
    dest=Path(__file__).with_name('piecewise_silu_fit.json')
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
