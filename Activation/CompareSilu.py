"""Compile the C++ SiLU comparison, run it, and plot the results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "exact_silu": "Exact SiLU",
    "taylor3": "Taylor-3 + linear bridge",
    "range_poly": "Range reduction + polynomial",
    "fast_sigmoid": "Fast sigmoid",
    "hard_silu": "Hard-SiLU",
    "piecewise_linear": "Piecewise linear",
    "lut_21": "LUT (21 values)",
    "relu": "ReLU",
}


def compile_and_run(args: argparse.Namespace, values_path: Path, timings_path: Path) -> None:
    source = Path(__file__).with_name("comparesilu.cpp")
    with tempfile.TemporaryDirectory(prefix="comparesilu-") as temporary_directory:
        executable = Path(temporary_directory) / "comparesilu"
        subprocess.run(
            [
                args.compiler,
                "-O3",
                "-ffast-math",
                "-march=native",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                str(source),
                "-o",
                str(executable),
                "-lm",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(executable),
                str(values_path),
                str(timings_path),
                str(args.minimum),
                str(args.maximum),
                str(args.samples),
                str(args.benchmark_elements),
                str(args.benchmark_warmup),
                str(args.benchmark_repeats),
            ],
            check=True,
        )


def read_timings(path: Path) -> dict[str, float]:
    with path.open(newline="") as file:
        return {row["key"]: float(row["median_ms"]) for row in csv.DictReader(file)}


def plot(values_path: Path, timings_path: Path, output: Path, show: bool) -> None:
    data = np.genfromtxt(values_path, delimiter=",", names=True)
    x = data["x"]
    reference = data["exact_silu"]
    timings = read_timings(timings_path)

    figure = plt.figure(figsize=(12, 13))
    grid = figure.add_gridspec(3, 1, height_ratios=(1.0, 1.0, 0.9))
    curve_axis = figure.add_subplot(grid[0])
    error_axis = figure.add_subplot(grid[1], sharex=curve_axis)
    timing_axis = figure.add_subplot(grid[2])

    curve_axis.plot(x, reference, color="black", linewidth=2.8, label="Exact SiLU")
    visible_limit = max(10.0, float(np.max(np.abs(reference))) + 2.0)
    for key, label in LABELS.items():
        if key == "exact_silu":
            continue
        approximation = data[key]
        visible = np.where(np.abs(approximation) <= visible_limit, approximation, np.nan)
        curve_axis.plot(x, visible, linewidth=1.5, label=label)
        error = np.abs(approximation - reference)
        finite = np.isfinite(error)
        error_axis.semilogy(
            x[finite],
            np.maximum(error[finite], 1e-9),
            linewidth=1.4,
            label=label,
        )

    curve_axis.set_title("SiLU and replacement activation functions (computed in C++)")
    curve_axis.set_ylabel("Output")
    curve_axis.grid(True, alpha=0.25)
    curve_axis.legend(ncol=2, fontsize=9)

    error_axis.set_title("Absolute error relative to exact SiLU")
    error_axis.set_xlabel("Input x")
    error_axis.set_ylabel("Absolute error (log scale)")
    error_axis.grid(True, which="both", alpha=0.25)

    keys = list(LABELS)
    timing_values = [timings[key] for key in keys]
    bars = timing_axis.barh([LABELS[key] for key in keys], timing_values, color="steelblue")
    timing_axis.invert_yaxis()
    timing_axis.set_title("Optimized C++ execution time (median after warm-up)")
    timing_axis.set_xlabel("Milliseconds per call (lower is better)")
    timing_axis.grid(True, axis="x", alpha=0.25)
    timing_axis.bar_label(bars, fmt="%.3f ms", padding=4, fontsize=9)
    timing_axis.set_xlim(0.0, max(timing_values) * 1.18)

    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    directory = Path(__file__).resolve().parent
    values_path = directory / "silu_comparison.csv"
    timings_path = directory / "silu_timings.csv"
    output = args.output.expanduser().resolve()

    compile_and_run(args, values_path, timings_path)
    plot(values_path, timings_path, output, args.show)
    print(f"Plot saved to: {output}")


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--minimum", type=float, default=-8.0)
    parser.add_argument("--maximum", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=4001)
    parser.add_argument("--benchmark-elements", type=int, default=1_000_000)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=directory / "silu_comparison.png")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
