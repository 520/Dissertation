"""Plot the selected adaptive SiLU LUT, its knots, and approximation error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit


DIRECTORY = Path(__file__).resolve().parent


def silu(x: np.ndarray) -> np.ndarray:
    return x * expit(x)


def approximate(x: np.ndarray, knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    predicted = np.interp(x, knots, values)
    predicted = np.where(x <= knots[0], 0.0, predicted)
    return np.where(x >= knots[-1], x, predicted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DIRECTORY / "adaptive_silu_lut.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DIRECTORY / "adaptive_silu_lut.png",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads(args.input.read_text())
    selected = result.get("selected")
    if selected is None:
        raise RuntimeError(f"No selected LUT candidate in {args.input}")

    knots = np.asarray(selected["knots"], dtype=np.float64)
    values = np.asarray(selected["values"], dtype=np.float64)
    evaluation_minimum, evaluation_maximum = result["evaluation_range"]
    x = np.linspace(evaluation_minimum, evaluation_maximum, 20_001)
    reference = silu(x)
    predicted = approximate(x, knots, values)
    error = np.abs(predicted - reference)

    figure, (curve_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)},
        layout="constrained",
    )

    curve_axis.plot(x, reference, color="black", linewidth=2.0, label="Exact SiLU")
    curve_axis.plot(
        x,
        predicted,
        color="#1473E6",
        linewidth=1.8,
        linestyle="--",
        label=f"Adaptive LUT ({selected['points']} points)",
    )
    curve_axis.scatter(
        knots,
        values,
        s=58,
        color="#E4572E",
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label="Optimised knots",
    )
    for index, (knot, value) in enumerate(zip(knots, values)):
        offset = 10 if index % 2 == 0 else -17
        curve_axis.annotate(
            f"{knot:.6f}",
            (knot, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=7.5,
            color="#8C2F16",
        )

    curve_axis.axvline(knots[0], color="#E4572E", alpha=0.25, linewidth=1.0)
    curve_axis.axvline(knots[-1], color="#E4572E", alpha=0.25, linewidth=1.0)
    curve_axis.set_ylabel("Output")
    curve_axis.margins(y=0.08)
    curve_axis.set_title(
        "Adaptive Non-uniform SiLU LUT with Optimised Knot Locations\n"
        f"MAE={selected['mae']:.6f}, RMSE={selected['rmse']:.6f}, "
        f"max error={selected['max_error']:.6f}"
    )
    curve_axis.legend(loc="upper left")
    curve_axis.grid(alpha=0.22)

    error_axis.plot(x, error, color="#7A5195", linewidth=1.5)
    error_axis.axhline(
        result["limits"]["max_error"],
        color="#C44E52",
        linestyle=":",
        linewidth=1.4,
        label=f"Max-error limit ({result['limits']['max_error']:.6f})",
    )
    error_axis.scatter(
        knots,
        np.zeros_like(knots),
        s=26,
        color="#E4572E",
        zorder=5,
        label="Knots",
    )
    error_axis.set_xlabel("Input")
    error_axis.set_ylabel("Absolute error")
    error_axis.grid(alpha=0.22)
    error_axis.legend(loc="upper right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)
    print(f"Plot: {args.output}")


if __name__ == "__main__":
    main()
