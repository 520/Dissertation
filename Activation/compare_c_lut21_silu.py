"""Compare torch.nn.functional.silu with an optimized C uniform LUT21."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import include_paths, library_paths


DIRECTORY = Path(__file__).resolve().parent
SOURCE = DIRECTORY / "uniform_lut21_silu.c"
CPP_SOURCE = DIRECTORY / "uniform_lut21_torch.cpp"
OPTIMIZATION_FLAGS = ["-O3", "-ffast-math", "-march=native"]


def compile_library(compiler: str, output: Path) -> list[str]:
    command = [
        compiler,
        *OPTIMIZATION_FLAGS,
        "-std=c11",
        "-Wall",
        "-Wextra",
    ]
    if platform.system() == "Darwin":
        command += ["-dynamiclib", str(SOURCE), "-o", str(output), "-lm"]
    else:
        command += ["-shared", "-fPIC", str(SOURCE), "-o", str(output), "-lm"]
    subprocess.run(command, check=True)
    return command


def compile_torch_cpp_extension(compiler: str, output: Path,
                                verbose: bool) -> list[str]:
    command = [compiler, *OPTIMIZATION_FLAGS, "-std=c++17", "-fPIC"]
    for include_directory in include_paths():
        command += ["-I", include_directory]
    command.append(str(CPP_SOURCE))

    if platform.system() == "Darwin":
        # PyTorch is already loaded by Python, so resolve its symbols at load time.
        torch_library_directory = library_paths()[0]
        command += [
            "-Xpreprocessor",
            "-fopenmp",
            "-L",
            torch_library_directory,
            f"-Wl,-rpath,{torch_library_directory}",
            "-lomp",
            "-dynamiclib",
            "-undefined",
            "dynamic_lookup",
            "-o",
            str(output),
        ]
    else:
        command.append("-fopenmp")
        for library_directory in library_paths():
            command += ["-L", library_directory, f"-Wl,-rpath,{library_directory}"]
        command += ["-shared", "-ltorch", "-ltorch_cpu", "-lc10", "-o", str(output)]

    if verbose:
        print("C++ compile:", " ".join(command))
    subprocess.run(command, check=True)
    torch.ops.load_library(str(output))
    return command


def load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    array_pointer = np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED")
    )
    library.lut21_silu_f32.argtypes = [
        array_pointer,
        array_pointer,
        ctypes.c_size_t,
    ]
    library.lut21_silu_f32.restype = None
    library.initialize_lut21.argtypes = []
    library.initialize_lut21.restype = None
    library.initialize_lut21()
    return library


def run_kernel(function, inputs: np.ndarray, output: np.ndarray) -> None:
    function(inputs, output, inputs.size)


def summarize(samples_ms: list[float], checksum: float,
              elements: int) -> dict[str, float]:
    median_ms = statistics.median(samples_ms)
    return {
        "median_ms": median_ms,
        "best_ms": min(samples_ms),
        "p10_ms": float(np.percentile(samples_ms, 10)),
        "p90_ms": float(np.percentile(samples_ms, 90)),
        "ns_per_element": median_ms * 1e6 / elements,
        "million_elements_per_second": elements / (median_ms * 1000.0),
        "checksum": checksum,
    }


def benchmark_c_lut(library: ctypes.CDLL, inputs: np.ndarray, warmup: int,
                    repeats: int) -> dict[str, float]:
    output = np.empty_like(inputs)
    for _ in range(warmup):
        run_kernel(library.lut21_silu_f32, inputs, output)

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        run_kernel(library.lut21_silu_f32, inputs, output)
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)
    return summarize(
        samples_ms, float(output.sum(dtype=np.float64)), inputs.size
    )


def benchmark_torch_silu(inputs: torch.Tensor, threads: int, warmup: int,
                         repeats: int) -> dict[str, float]:
    torch.set_num_threads(threads)
    with torch.inference_mode():
        for _ in range(warmup):
            output = F.silu(inputs)

        samples_ms: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            output = F.silu(inputs)
            samples_ms.append((time.perf_counter_ns() - start) / 1e6)
    return summarize(samples_ms, float(output.double().sum()), inputs.numel())


def benchmark_torch_cpp_lut(inputs: torch.Tensor, threads: int, warmup: int,
                            repeats: int) -> dict[str, float]:
    torch.set_num_threads(threads)
    operation = torch.ops.dissertation_lut21.forward
    with torch.inference_mode():
        for _ in range(warmup):
            output = operation(inputs)

        samples_ms: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            output = operation(inputs)
            samples_ms.append((time.perf_counter_ns() - start) / 1e6)
    return summarize(samples_ms, float(output.double().sum()), inputs.numel())


def measure_error(library: ctypes.CDLL, minimum: float, maximum: float,
                  samples: int) -> dict[str, object]:
    x = np.linspace(minimum, maximum, samples, dtype=np.float32)
    c_approximation = np.empty_like(x)
    run_kernel(library.lut21_silu_f32, x, c_approximation)

    with torch.inference_mode():
        torch_input = torch.from_numpy(x)
        reference = F.silu(torch_input).numpy().astype(np.float64)
        cpp_approximation = (
            torch.ops.dissertation_lut21.forward(torch_input).numpy()
        )

    def statistics_for(approximation: np.ndarray) -> dict[str, float]:
        difference = approximation.astype(np.float64) - reference
        absolute = np.abs(difference)
        maximum_index = int(np.argmax(absolute))
        return {
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "max_absolute_error": float(absolute[maximum_index]),
            "max_error_at_x": float(x[maximum_index]),
        }

    return {
        "c_lut21": statistics_for(c_approximation),
        "torch_cpp_lut21": statistics_for(cpp_approximation),
        "cpp_vs_c_max_absolute_difference": float(
            np.max(np.abs(cpp_approximation - c_approximation))
        ),
    }


def main(args: argparse.Namespace) -> None:
    original_torch_threads = torch.get_num_threads()
    requested_torch_threads = (
        original_torch_threads if args.torch_threads == 0 else args.torch_threads
    )
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    with tempfile.TemporaryDirectory(prefix="c-lut21-") as temporary_directory:
        library_path = Path(temporary_directory) / f"liblut21{suffix}"
        compile_command = compile_library(args.compiler, library_path)
        library = load_library(library_path)
        cpp_library_path = Path(temporary_directory) / f"libtorch_lut21{suffix}"
        cpp_compile_command = compile_torch_cpp_extension(
            args.cxx, cpp_library_path, args.verbose_build
        )

        rng = np.random.default_rng(args.seed)
        if args.distribution == "uniform":
            inputs = rng.uniform(
                args.minimum, args.maximum, args.elements
            ).astype(np.float32)
            distribution_description = (
                f"uniform(min={args.minimum:g}, max={args.maximum:g})"
            )
        else:
            inputs = rng.normal(
                0.0, args.input_std, args.elements
            ).astype(np.float32)
            distribution_description = f"normal(mean=0, std={args.input_std:g})"
        torch_inputs = torch.from_numpy(inputs)

        lut_timing = benchmark_c_lut(
            library, inputs, args.warmup, args.repeats
        )
        torch_silu_1t = benchmark_torch_silu(
            torch_inputs, 1, args.warmup, args.repeats
        )
        cpp_lut_1t = benchmark_torch_cpp_lut(
            torch_inputs, 1, args.warmup, args.repeats
        )
        if requested_torch_threads == 1:
            torch_silu_mt = torch_silu_1t
            cpp_lut_mt = cpp_lut_1t
        else:
            torch_silu_mt = benchmark_torch_silu(
                torch_inputs, requested_torch_threads, args.warmup, args.repeats
            )
            cpp_lut_mt = benchmark_torch_cpp_lut(
                torch_inputs, requested_torch_threads, args.warmup, args.repeats
            )
        error = measure_error(library, args.minimum, args.maximum, args.samples)
    torch.set_num_threads(original_torch_threads)

    results = {
        "compiler_command": compile_command,
        "cpp_compiler_command": cpp_compile_command,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "input": {
            "elements": args.elements,
            "distribution": distribution_description,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "timing": {
            "c_uniform_lut21_single_thread": lut_timing,
            "torch_cpp_lut21_1_thread": cpp_lut_1t,
            "torch_functional_silu_1_thread": torch_silu_1t,
            f"torch_cpp_lut21_{requested_torch_threads}_threads": cpp_lut_mt,
            f"torch_functional_silu_{requested_torch_threads}_threads": torch_silu_mt,
        },
        "speedup": {
            "c_lut21_over_torch_silu_1_thread": (
                torch_silu_1t["median_ms"] / lut_timing["median_ms"]
            ),
            f"c_lut21_over_torch_silu_{requested_torch_threads}_threads": (
                torch_silu_mt["median_ms"] / lut_timing["median_ms"]
            ),
            "cpp_lut21_over_torch_silu_1_thread": (
                torch_silu_1t["median_ms"] / cpp_lut_1t["median_ms"]
            ),
            f"cpp_lut21_over_torch_silu_{requested_torch_threads}_threads": (
                torch_silu_mt["median_ms"] / cpp_lut_mt["median_ms"]
            ),
        },
        "lut21_error_vs_torch_functional_silu": {
            "range": [args.minimum, args.maximum],
            "samples": args.samples,
            **error,
        },
    }

    output_path = args.output.expanduser().resolve()
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("Compile:", " ".join(compile_command))
    print(f"C uniform LUT21 (1 thread):       {lut_timing['median_ms']:.6f} ms")
    print(f"PyTorch C++ LUT21 (1 thread):     {cpp_lut_1t['median_ms']:.6f} ms")
    print(f"torch F.silu (1 thread):          {torch_silu_1t['median_ms']:.6f} ms")
    print(
        f"PyTorch C++ LUT21 ({requested_torch_threads} threads):"
        f" {cpp_lut_mt['median_ms']:.6f} ms"
    )
    print(
        f"torch F.silu ({requested_torch_threads} threads):"
        f"        {torch_silu_mt['median_ms']:.6f} ms"
    )
    print(
        "C LUT21 speedup vs torch 1 thread: "
        f"{torch_silu_1t['median_ms'] / lut_timing['median_ms']:.3f}x"
    )
    print(
        f"C LUT21 speedup vs torch {requested_torch_threads} threads: "
        f"{torch_silu_mt['median_ms'] / lut_timing['median_ms']:.3f}x"
    )
    print(
        f"C++ LUT21 speedup vs torch {requested_torch_threads} threads: "
        f"{torch_silu_mt['median_ms'] / cpp_lut_mt['median_ms']:.3f}x"
    )
    c_error = error["c_lut21"]
    cpp_error = error["torch_cpp_lut21"]
    print(f"C LUT21 MAE:             {c_error['mae']:.8f}")
    print(f"C++ LUT21 MAE:           {cpp_error['mae']:.8f}")
    print(f"C++ vs C max difference: {error['cpp_vs_c_max_absolute_difference']:.9g}")
    print(
        f"C++ LUT21 max abs error: {cpp_error['max_absolute_error']:.8f} "
        f"at x={cpp_error['max_error_at_x']:.6f}"
    )
    print(f"Results:             {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--elements", type=int, default=1_000_000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="PyTorch multithread count; 0 keeps its current default",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--distribution",
        choices=("uniform", "normal"),
        default="uniform",
        help="Benchmark input distribution; uniform matches comparesilu.cpp",
    )
    parser.add_argument("--input-std", type=float, default=2.0)
    parser.add_argument("--minimum", type=float, default=-8.0)
    parser.add_argument("--maximum", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=4001)
    parser.add_argument(
        "--output", type=Path, default=DIRECTORY / "c_lut21_vs_silu_results.json"
    )
    parser.add_argument("--verbose-build", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
