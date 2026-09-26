#!/usr/bin/env python3
"""Run the KITTI-style C++ activation ablation on the VOC YOLOv8n model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import compare_yolo_activations_kitti as experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
experiment.DEFAULT_MODEL = (
    PROJECT_ROOT / "original/yolov8n_voc/final_model_factorized_lwi.pt"
)
experiment.DEFAULT_DATA = PROJECT_ROOT / "datasets/VOC/VOC.yaml"
experiment.DEFAULT_RESULTS = (
    PROJECT_ROOT / "Activation/yolo_voc_activation_ablation.json"
)
experiment.DEFAULT_DATASET_LABEL = "VOC"


def run_parallel(worker_count: int) -> None:
    args = experiment.parse_args()
    if args.only or args.skip_latency or args.skip_validation:
        raise ValueError("Parallel VOC ablation requires all activations and both metrics")
    if worker_count < 2:
        raise ValueError("--parallel-validation must be at least 2")

    variants = [*experiment.CPP_ACTIVATIONS, "torch_cpp_lut21"]
    groups = [variants[index::worker_count] for index in range(worker_count)]
    groups = [group for group in groups if group]
    common = [
        "--data", str(args.data),
        "--dataset-label", args.dataset_label,
        "--imgsz", str(args.imgsz), "--batch", str(args.batch),
        "--threads", str(args.threads), "--workers", str(args.workers),
        "--latency-warmup", str(args.latency_warmup),
        "--latency-max-images", str(args.latency_max_images),
        "--cxx", args.cxx,
    ]
    if args.model is None:
        common.extend(["--git-object", args.git_object])
    else:
        common.extend(["--model", str(args.model)])
    if args.verbose_build:
        common.append("--verbose-build")

    with tempfile.TemporaryDirectory(prefix="voc-activation-ablation-") as temp:
        temporary = Path(temp)

        def validate_group(index: int, group: list[str]) -> Path:
            output = temporary / f"validation-{index}.json"
            log = temporary / f"validation-{index}.log"
            command = [
                sys.executable, str(Path(__file__).resolve()), *common,
                "--skip-latency", "--only", *group,
                "--results", str(output),
            ]
            with log.open("w") as stream:
                completed = subprocess.run(
                    command, cwd=PROJECT_ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, check=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"VOC validation group {index} failed:\n"
                    + log.read_text(errors="replace")[-4000:]
                )
            return output

        print(f"Validating {len(variants)} variants in {len(groups)} groups", flush=True)
        shard_paths: list[Path] = []
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = {
                pool.submit(validate_group, index, group): index
                for index, group in enumerate(groups)
            }
            for future in as_completed(futures):
                shard_paths.append(future.result())
                print(f"Validation group {futures[future] + 1} complete", flush=True)

        latency_path = temporary / "latency.json"
        latency_command = [
            sys.executable, str(Path(__file__).resolve()), *common,
            "--skip-validation", "--results", str(latency_path),
        ]
        print("Measuring forward latency without competing workers", flush=True)
        subprocess.run(latency_command, cwd=PROJECT_ROOT, check=True)
        combined = json.loads(latency_path.read_text())
        validation_key = f"{args.dataset_label.lower()}_validation"
        for path in shard_paths:
            shard = json.loads(path.read_text())
            for name, entry in shard["comparisons"].items():
                combined["comparisons"][name][validation_key] = entry[validation_key]

        expected = {"native_silu", *variants}
        if set(combined["comparisons"]) != expected:
            raise RuntimeError("Incomplete activation comparison after merging")
        if any(validation_key not in entry for entry in combined["comparisons"].values()):
            raise RuntimeError("Missing VOC validation result after merging")

        native = combined["comparisons"]["native_silu"][validation_key]
        for name, entry in combined["comparisons"].items():
            if name == "native_silu":
                continue
            relative = combined["relative_to_native_silu"][name]
            for metric in ("map50", "map50_95"):
                relative[f"{metric}_absolute_change"] = (
                    entry[validation_key][metric] - native[metric]
                )
        combined["validation_execution"] = (
            f"{len(groups)} independent CPU processes, one thread each; "
            "forward latency measured separately without competing workers"
        )
        output = args.results.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(combined, indent=2) + "\n")
        print(f"Saved results: {output}", flush=True)


if __name__ == "__main__":
    parallel_parser = argparse.ArgumentParser(add_help=False)
    parallel_parser.add_argument("--parallel-validation", type=int, default=1)
    parallel_args, remaining = parallel_parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if parallel_args.parallel_validation == 1:
        experiment.main()
    else:
        run_parallel(parallel_args.parallel_validation)
