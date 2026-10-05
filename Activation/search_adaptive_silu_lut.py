"""Search the smallest asymmetric, non-uniform piecewise-linear SiLU LUT.

The interpolation range is learned instead of being fixed to [-5, 5].  The
search starts with a small LUT and adds one knot at a time until all three
requested error limits are met.  Negative and positive knot locations and tail
thresholds are optimised independently.  Zero remains a knot because it is an
exact, natural boundary for SiLU.  Outside the learned range the approximation
uses SiLU's asymptotes: zero on the negative side and the identity on the
positive side.

MAE and RMSE require a probability measure, so by default they are measured on
a uniform grid over [-8, 8], matching ``comparesilu.cpp``.  The maximum error
is additionally checked over every interpolation segment and both infinite
tails.

This is a deterministic numerical search, not a proof of global optimality.
Run it with the project's SciPy-enabled Python environment, for example:

    python Activation/search_adaptive_silu_lut.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.special import expit


DIRECTORY = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Limits:
    mae: float
    rmse: float
    max_error: float


@dataclass
class Candidate:
    points: int
    interpolation_segments: int
    negative_points: int
    positive_points: int
    minimum: float
    maximum: float
    knots: list[float]
    values: list[float]
    mae: float
    rmse: float
    max_error: float
    worst_input: float
    normalized_score: float
    passes: bool
    optimizer_success: bool
    optimizer_message: str
    seed: int


def silu(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable SiLU reference."""
    return np.asarray(x) * expit(x)


def knots_from_gaps(gaps: Sequence[float], negative_segments: int) -> np.ndarray:
    """Build ordered knots with independently spaced negative/positive sides."""
    gaps = np.asarray(gaps, dtype=np.float64)
    negative = -np.cumsum(gaps[:negative_segments])[::-1]
    positive = np.cumsum(gaps[negative_segments:])
    return np.r_[negative, 0.0, positive]


def approximate(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Evaluate the asymmetric interpolating LUT with zero/identity tails."""
    predicted = np.interp(x, knots, silu(knots))
    predicted = np.where(x <= knots[0], 0.0, predicted)
    return np.where(x >= knots[-1], x, predicted)


def grid_metrics(
    knots: np.ndarray,
    x: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, float, float]:
    error = np.abs(approximate(x, knots) - reference)
    worst_index = int(np.argmax(error))
    return (
        float(np.mean(error)),
        float(np.sqrt(np.mean(np.square(error)))),
        float(error[worst_index]),
        float(x[worst_index]),
    )


def continuous_max_error(knots: np.ndarray) -> tuple[float, float]:
    """Check segment interiors and both infinite tails for missed error peaks."""
    values = silu(knots)
    best_error = 0.0
    best_x = 0.0

    def consider(x_value: float, predicted: float) -> None:
        nonlocal best_error, best_x
        error = abs(predicted - float(silu(x_value)))
        if error > best_error:
            best_error = error
            best_x = x_value

    for left, right, y_left, y_right in zip(
        knots[:-1],
        knots[1:],
        values[:-1],
        values[1:],
    ):
        slope = float((y_right - y_left) / (right - left))
        intercept = float(y_left - slope * left)
        consider(float(left), slope * float(left) + intercept)
        consider(float(right), slope * float(right) + intercept)
        # Splitting makes the bounded scalar search robust to multiple extrema.
        edges = np.linspace(left, right, 33)
        for sub_left, sub_right in zip(edges[:-1], edges[1:]):
            optimum = minimize_scalar(
                lambda z: -abs(slope * z + intercept - float(silu(z))),
                bounds=(float(sub_left), float(sub_right)),
                method="bounded",
            )
            consider(float(optimum.x), slope * float(optimum.x) + intercept)

    maximum = float(knots[-1])
    tail_right = max(20.0, maximum + 1.0)
    tail_edges = np.linspace(maximum, tail_right, 65)
    for sub_left, sub_right in zip(tail_edges[:-1], tail_edges[1:]):
        optimum = minimize_scalar(
            lambda z: -abs(z - float(silu(z))),
            bounds=(float(sub_left), float(sub_right)),
            method="bounded",
        )
        consider(float(optimum.x), float(optimum.x))

    minimum = float(knots[0])
    tail_left = min(-20.0, minimum - 1.0)
    tail_edges = np.linspace(tail_left, minimum, 65)
    for sub_left, sub_right in zip(tail_edges[:-1], tail_edges[1:]):
        optimum = minimize_scalar(
            lambda z: -abs(float(silu(z))),
            bounds=(float(sub_left), float(sub_right)),
            method="bounded",
        )
        consider(float(optimum.x), 0.0)

    return best_error, best_x


def normalized_score(metrics: Sequence[float], limits: Limits) -> float:
    ratios = np.asarray(metrics, dtype=np.float64) / np.asarray(
        [limits.mae, limits.rmse, limits.max_error], dtype=np.float64
    )
    # The maximum ratio enforces all constraints.  The small mean term breaks
    # ties in favour of candidates that improve all three errors.
    return float(np.max(ratios) + 0.01 * np.mean(ratios))


def search_point_count(
    points: int,
    search_x: np.ndarray,
    search_reference: np.ndarray,
    limits: Limits,
    seeds: Sequence[int],
    maxiter: int,
    popsize: int,
    minimum_gap: float,
    maximum_gap: float,
) -> tuple[np.ndarray, object, int, int]:
    # For even point counts one side necessarily gets one more segment.  The
    # mirrored allocation has the same score on the symmetric evaluation grid.
    negative_segments = (points - 1) // 2
    total_segments = points - 1

    def objective(gaps: np.ndarray) -> float:
        knots = knots_from_gaps(gaps, negative_segments)
        mae, rmse, max_error, _ = grid_metrics(knots, search_x, search_reference)
        return normalized_score((mae, rmse, max_error), limits)

    best_fit = None
    best_seed = -1
    for seed in seeds:
        fit = differential_evolution(
            objective,
            bounds=[(minimum_gap, maximum_gap)] * total_segments,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-8,
            polish=True,
            updating="immediate",
            workers=1,
        )
        if best_fit is None or fit.fun < best_fit.fun:
            best_fit = fit
            best_seed = seed

    assert best_fit is not None
    return (
        knots_from_gaps(best_fit.x, negative_segments),
        best_fit,
        best_seed,
        negative_segments,
    )


def verify_candidate(
    points: int,
    knots: np.ndarray,
    fit: object,
    seed: int,
    negative_segments: int,
    verification_x: np.ndarray,
    verification_reference: np.ndarray,
    limits: Limits,
) -> Candidate:
    mae, rmse, dense_max, dense_worst_x = grid_metrics(
        knots, verification_x, verification_reference
    )
    extrema_max, extrema_worst_x = continuous_max_error(knots)
    if extrema_max > dense_max:
        max_error = extrema_max
        worst_input = extrema_worst_x
    else:
        max_error = dense_max
        worst_input = dense_worst_x

    metrics = (mae, rmse, max_error)
    passes = (
        mae < limits.mae
        and rmse < limits.rmse
        and max_error < limits.max_error
    )
    return Candidate(
        points=points,
        interpolation_segments=points - 1,
        negative_points=negative_segments,
        positive_points=points - negative_segments - 1,
        minimum=float(knots[0]),
        maximum=float(knots[-1]),
        knots=[float(value) for value in knots],
        values=[float(value) for value in silu(knots)],
        mae=mae,
        rmse=rmse,
        max_error=max_error,
        worst_input=worst_input,
        normalized_score=normalized_score(metrics, limits),
        passes=passes,
        optimizer_success=bool(fit.success),
        optimizer_message=str(fit.message),
        seed=seed,
    )


def write_csv(path: Path, candidates: Sequence[Candidate]) -> None:
    fields = [
        "points",
        "interpolation_segments",
        "negative_points",
        "positive_points",
        "minimum",
        "maximum",
        "mae",
        "rmse",
        "max_error",
        "worst_input",
        "normalized_score",
        "passes",
        "seed",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            row = asdict(candidate)
            writer.writerow({field: row[field] for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mae-limit", type=float, default=0.00642035)
    parser.add_argument("--rmse-limit", type=float, default=0.00999424)
    parser.add_argument("--max-error-limit", type=float, default=0.0334644)
    parser.add_argument("--evaluation-minimum", type=float, default=-8.0)
    parser.add_argument("--evaluation-maximum", type=float, default=8.0)
    parser.add_argument("--minimum-points", type=int, default=3)
    parser.add_argument(
        "--maximum-points",
        type=int,
        default=0,
        help="Optional safety cap; zero means keep adding points until a fit passes.",
    )
    parser.add_argument("--search-samples", type=int, default=50_001)
    parser.add_argument("--verification-samples", type=int, default=1_000_001)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-gap", type=float, default=0.05)
    parser.add_argument("--maximum-gap", type=float, default=3.0)
    parser.add_argument(
        "--continue-after-pass",
        action="store_true",
        help="Continue to --maximum-points to produce a longer Pareto table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DIRECTORY / "adaptive_silu_lut.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DIRECTORY / "adaptive_silu_pareto.csv",
    )
    args = parser.parse_args()

    limits = (args.mae_limit, args.rmse_limit, args.max_error_limit)
    if any(value <= 0.0 for value in limits):
        parser.error("All error limits must be positive")
    if args.evaluation_maximum <= args.evaluation_minimum:
        parser.error("evaluation-maximum must exceed evaluation-minimum")
    if args.search_samples < 101 or args.verification_samples < 1001:
        parser.error("Use at least 101 search samples and 1001 verification samples")
    if args.minimum_gap <= 0.0 or args.maximum_gap <= args.minimum_gap:
        parser.error("Gap bounds must satisfy 0 < minimum-gap < maximum-gap")
    if args.maxiter < 1 or args.popsize < 1 or args.restarts < 1:
        parser.error("maxiter, popsize and restarts must be positive")

    args.minimum_points = max(3, args.minimum_points)
    if args.maximum_points and args.maximum_points < args.minimum_points:
        parser.error("maximum-points must be zero or at least minimum-points")
    if args.continue_after_pass and args.maximum_points == 0:
        parser.error("--continue-after-pass requires a finite --maximum-points")
    return args


def main() -> None:
    args = parse_args()
    limits = Limits(args.mae_limit, args.rmse_limit, args.max_error_limit)
    search_x = np.linspace(
        args.evaluation_minimum,
        args.evaluation_maximum,
        args.search_samples,
        dtype=np.float64,
    )
    verification_x = np.linspace(
        args.evaluation_minimum,
        args.evaluation_maximum,
        args.verification_samples,
        dtype=np.float64,
    )
    search_reference = silu(search_x)
    verification_reference = silu(verification_x)
    seeds = [args.seed + index for index in range(args.restarts)]

    candidates: list[Candidate] = []
    selected: Candidate | None = None
    points = args.minimum_points
    while args.maximum_points == 0 or points <= args.maximum_points:
        print(f"Searching {points} points ({points - 1} interpolation segments) ...")
        knots, fit, seed, negative_segments = search_point_count(
            points=points,
            search_x=search_x,
            search_reference=search_reference,
            limits=limits,
            seeds=seeds,
            maxiter=args.maxiter,
            popsize=args.popsize,
            minimum_gap=args.minimum_gap,
            maximum_gap=args.maximum_gap,
        )
        candidate = verify_candidate(
            points,
            knots,
            fit,
            seed,
            negative_segments,
            verification_x,
            verification_reference,
            limits,
        )
        candidates.append(candidate)
        print(
            f"  range=[{candidate.minimum:.9g}, {candidate.maximum:.9g}] "
            f"MAE={candidate.mae:.9g} RMSE={candidate.rmse:.9g} "
            f"max={candidate.max_error:.9g} pass={candidate.passes}"
        )
        if candidate.passes and selected is None:
            selected = candidate
            if not args.continue_after_pass:
                break
        points += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "description": (
            "Adaptive asymmetric non-uniform interpolating SiLU LUT with "
            "independently learned negative/positive ranges and zero/identity tails"
        ),
        "qualification": (
            "Seeded numerical candidate; smallest passing searched point count, "
            "not a proof of the global optimum"
        ),
        "limits": asdict(limits),
        "metric_distribution": "uniform grid",
        "evaluation_range": [args.evaluation_minimum, args.evaluation_maximum],
        "search_samples": args.search_samples,
        "verification_samples": args.verification_samples,
        "seeds": seeds,
        "selected": asdict(selected) if selected is not None else None,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(args.csv_output, candidates)

    print(f"JSON: {args.output}")
    print(f"CSV:  {args.csv_output}")
    if selected is None:
        raise SystemExit("No candidate met all error limits")
    print(
        f"Selected {selected.points} points / "
        f"{selected.interpolation_segments} interpolation segments."
    )


if __name__ == "__main__":
    main()
