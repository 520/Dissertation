"""CPU latency baseline for an Ultralytics-exported TFLite model on the LiteRT XNNPACK delegate.

Protocol (mirrors tensorrt_baseline.ipynb): batch 1, static 640x640 input. The first --num-images (default 500)
sorted KITTI val images are letterboxed and preloaded, then cycled: 20 warmup invokes, then timed invokes until at
least --min-seconds have elapsed AND at least --min-iters samples (default: one full pass over all images) were
collected. Only ``interpreter.invoke()`` is timed; copying the next image into the input tensor happens outside the
timed region, so this is pure inference latency without pre/post-processing or NMS.

Each thread count is measured with XNNPACK (LiteRT default delegate, OpResolverType.AUTO) and, unless
--no-reference is given, with the built-in kernels only (BUILTIN_WITHOUT_DEFAULT_DELEGATES) as a reference.

Usage:
    .venv/Scripts/python.exe tflite_xnnpack_benchmark.py
    .venv/Scripts/python.exe tflite_xnnpack_benchmark.py --threads 1 4 8 --min-seconds 15
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert import __version__ as LITERT_VERSION
from ai_edge_litert.interpreter import Interpreter, OpResolverType

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "original/yolov8n_kitti/8n_float32_saved_model/8n_float32_float32.tflite"
DEFAULT_PT = ROOT / "original/yolov8n_kitti/8n_float32.pt"
DEFAULT_IMAGES = ROOT / "PTQ/datasets/kitti/images/val"
DEFAULT_OUT = ROOT / "tflite_xnnpack_baseline_results/8n_float32_fp32"

MODES = {
    "xnnpack": OpResolverType.AUTO,  # built-in ops + default delegates (XNNPACK on CPU)
    "builtin_no_xnnpack": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
}


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def letterbox(img: np.ndarray, size: int = 640, color: int = 114) -> np.ndarray:
    """Ultralytics-style letterbox (keep ratio, centre pad with 114)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = int(round((size - nh) / 2 - 0.1))
    left = int(round((size - nw) / 2 - 0.1))
    return cv2.copyMakeBorder(img, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT,
                              value=(color, color, color))


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def load_images(images_dir: Path, size: int, n: int) -> tuple[np.ndarray, dict]:
    """Return (N, H, W, 3) uint8 letterboxed RGB images (first n sorted files) and a description of the source.

    Images are kept as uint8 (~1.2 MB each) and converted to float32 [0, 1] just before each invoke.
    """
    files = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) \
        if images_dir.is_dir() else []
    files = files[:n]
    if not files:
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, (1, size, size, 3), dtype=np.uint8), {"source": "random uint8, seed 0", "count": 1}
    imgs = np.empty((len(files), size, size, 3), dtype=np.uint8)
    for i, f in enumerate(files):
        im = cv2.imread(str(f))
        if im is None:
            raise RuntimeError(f"cannot read {f}")
        imgs[i] = letterbox(im, size)[..., ::-1]  # BGR -> RGB
    return imgs, {"source": rel(images_dir), "count": len(files), "first": files[0].name, "last": files[-1].name}


def to_input(img_u8: np.ndarray, out: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 [0, 1] into the preallocated (1, H, W, 3) buffer."""
    np.divide(img_u8, np.float32(255.0), out=out[0], dtype=np.float32)
    return out


def make_interpreter(model: Path, threads: int, mode: str) -> Interpreter:
    it = Interpreter(model_path=str(model), num_threads=threads, experimental_op_resolver_type=MODES[mode])
    it.allocate_tensors()
    return it


def delegation_report(it: Interpreter) -> dict:
    """Which ops ended up inside XNNPACK partitions and which fell back to built-in CPU kernels."""
    ops = it._get_ops_details()
    delegates = [o for o in ops if o["op_name"] == "DELEGATE"]
    original = [o for o in ops if o["op_name"] != "DELEGATE"]
    if not delegates:
        return {"partitions": 0, "ops_total": len(original), "ops_delegated": 0,
                "non_delegated_ops": dict(collections.Counter(o["op_name"] for o in original))}

    producer = {int(t): o for o in original for t in o["outputs"]}
    delegate_out = {int(t) for d in delegates for t in d["outputs"]}
    graph_inputs = {int(d["index"]) for d in it.get_input_details()}
    needed = [int(t) for d in delegates for t in d["inputs"]] + [int(d["index"]) for d in it.get_output_details()]

    fallback, seen = {}, set()
    while needed:  # walk upwards from delegate inputs / graph outputs through ops not covered by a delegate
        t = needed.pop()
        if t in seen or t in delegate_out or t in graph_inputs or t not in producer:
            continue
        seen.add(t)
        op = producer[t]
        fallback[op["index"]] = op
        needed.extend(int(i) for i in op["inputs"])
    return {
        "partitions": len(delegates),
        "ops_total": len(original),
        "ops_delegated": len(original) - len(fallback),
        "non_delegated_ops": dict(collections.Counter(o["op_name"] for o in fallback.values())),
        "non_delegated_op_indices": sorted(int(i) for i in fallback),
    }


def stats(ms: np.ndarray) -> dict:
    return {
        "samples": int(ms.size),
        "mean_ms": float(ms.mean()),
        "median_ms": float(np.median(ms)),
        "p75_ms": float(np.percentile(ms, 75)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "std_ms": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
        "fps_inverse_mean_latency": float(1000.0 / ms.mean()),
    }


def benchmark(it: Interpreter, imgs: np.ndarray, warmup: int, min_seconds: float, min_iters: int) -> np.ndarray:
    """Cycle through imgs; time only invoke(). Warmup also cycles from image 0, timed run restarts at image 0."""
    inp = it.get_input_details()[0]["index"]
    buf = np.empty((1, *imgs.shape[1:]), dtype=np.float32)
    n = len(imgs)
    set_tensor, invoke = it.set_tensor, it.invoke
    for i in range(warmup):
        set_tensor(inp, to_input(imgs[i % n], buf))
        invoke()
    samples = []
    gc.collect()
    gc.disable()
    try:
        t_end = time.perf_counter() + min_seconds
        i = 0
        while i < min_iters or time.perf_counter() < t_end:
            set_tensor(inp, to_input(imgs[i % n], buf))  # untimed: next image into the input tensor
            t0 = time.perf_counter_ns()
            invoke()
            samples.append(time.perf_counter_ns() - t0)
            i += 1
    finally:
        gc.enable()
    return np.asarray(samples, dtype=np.float64) / 1e6


def torch_reference_check(pt: Path, x_nhwc: np.ndarray, y_tflite: np.ndarray) -> dict:
    """Compare TFLite output with the original PyTorch model on the same input.

    In Ultralytics 8.4.x only EdgeTPU exports normalise boxes; SavedModel/TFLite boxes are xywh in input pixels,
    the same as the PyTorch head, so outputs are compared directly.
    """
    import torch
    from ultralytics import YOLO

    model = YOLO(str(pt)).model.float().eval()
    with torch.no_grad():
        y = model(torch.from_numpy(x_nhwc).permute(0, 3, 1, 2).contiguous())
    y = (y[0] if isinstance(y, (list, tuple)) else y).numpy()
    boxes_diff = np.abs(y_tflite[:, :4] - y[:, :4])
    cls_diff = np.abs(y_tflite[:, 4:] - y[:, 4:])
    top_t = np.argsort(-y_tflite[0, 4:].max(0))[:100]
    top_p = np.argsort(-y[0, 4:].max(0))[:100]
    return {
        "output_shape": list(y.shape),
        "box_max_abs_diff_px": float(boxes_diff.max()),
        "box_mean_abs_diff_px": float(boxes_diff.mean()),
        "score_max_abs_diff": float(cls_diff.max()),
        "score_mean_abs_diff": float(cls_diff.mean()),
        "top100_anchor_overlap": int(len(set(top_t.tolist()) & set(top_p.tolist()))),
        "max_score_tflite": float(y_tflite[0, 4:].max()),
        "max_score_torch": float(y[0, 4:].max()),
    }


def write_markdown(path: Path, res: dict) -> None:
    env, d = res["environment"], res["delegation"]
    lines = [
        f"# TFLite FP32 CPU latency baseline — {res['model']['name']}",
        "",
        f"- Model: `{res['model']['path']}` ({res['model']['size_mb']:.2f} MB, input {res['model']['input_shape']} "
        f"{res['model']['input_dtype']}, output {res['model']['output_shape']})",
        f"- Runtime: LiteRT (ai-edge-litert) {env['ai_edge_litert']}, Python {env['python']}, {env['os']}",
        f"- CPU: {env['cpu']} ({env['logical_cpus']} logical CPUs)",
        f"- XNNPACK delegation: {d['ops_delegated']}/{d['ops_total']} ops in {d['partitions']} partitions; "
        f"not delegated: {d['non_delegated_ops'] or 'none'}",
        f"- Protocol: batch 1, {res['protocol']['images']['count']} images from `{res['protocol']['images']['source']}` "
        f"({res['protocol']['images'].get('first', '')} ... {res['protocol']['images'].get('last', '')}, letterboxed, "
        f"preloaded, cycled); {res['protocol']['warmup']} warmup invokes, then >= {res['protocol']['min_seconds']} s "
        f"and >= {res['protocol']['min_iters']} timed `invoke()` calls (every image timed at least once); "
        "input copy, pre/post-processing and NMS excluded.",
        "",
        "| Backend | Threads | Median (ms) | P75 (ms) | P95 (ms) | Mean (ms) | Std (ms) | Min (ms) | FPS (1/mean) | Samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in res["runs"]:
        s = r["latency"]
        lines.append(
            f"| {r['mode']} | {r['threads']} | {s['median_ms']:.2f} | {s['p75_ms']:.2f} | {s['p95_ms']:.2f} | "
            f"{s['mean_ms']:.2f} | {s['std_ms']:.2f} | {s['min_ms']:.2f} | {s['fps_inverse_mean_latency']:.1f} | "
            f"{s['samples']} |"
        )
    if res.get("numerical_check"):
        c = res["numerical_check"]
        lines += [
            "",
            "Numerical check vs PyTorch `.pt` (same input): "
            f"box max |diff| {c['box_max_abs_diff_px']:.4f} px, score max |diff| {c['score_max_abs_diff']:.2e}, "
            f"top-100 anchor overlap {c['top100_anchor_overlap']}/100.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--pt", type=Path, default=DEFAULT_PT, help="PyTorch weights for the numerical check")
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="directory of input images (sorted)")
    ap.add_argument("--num-images", type=int, default=500, help="use the first N sorted images, cycled")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--min-seconds", type=float, default=15.0)
    ap.add_argument("--min-iters", type=int, default=None, help="default: one full pass over the images")
    ap.add_argument("--no-reference", action="store_true", help="skip the built-in-kernels (no XNNPACK) runs")
    ap.add_argument("--no-check", action="store_true", help="skip the PyTorch numerical check")
    args = ap.parse_args()

    model = args.model.resolve()
    probe = make_interpreter(model, 1, "xnnpack")
    ind, outd = probe.get_input_details()[0], probe.get_output_details()[0]
    imgsz = int(ind["shape"][1])
    t0 = time.perf_counter()
    imgs, img_info = load_images(args.images, imgsz, args.num_images)
    print(f"Loaded {img_info['count']} images from {img_info['source']} in {time.perf_counter() - t0:.1f}s "
          f"({imgs.nbytes / 1e6:.0f} MB uint8)")
    min_iters = args.min_iters if args.min_iters is not None else len(imgs)
    delegation = delegation_report(probe)
    if delegation["partitions"] == 0:
        sys.exit("XNNPACK delegate was not applied - aborting (check the LiteRT build).")
    x = to_input(imgs[0], np.empty((1, imgsz, imgsz, 3), dtype=np.float32)).copy()  # first image, numerical check
    probe.set_tensor(ind["index"], x)
    probe.invoke()
    y_tflite = probe.get_tensor(outd["index"]).copy()
    del probe

    res = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "name": model.stem,
            "path": rel(model),
            "size_mb": model.stat().st_size / 1e6,
            "sha256": sha256(model),
            "input_shape": [int(v) for v in ind["shape"]],
            "input_dtype": np.dtype(ind["dtype"]).name,
            "output_shape": [int(v) for v in outd["shape"]],
        },
        "environment": {
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "python": platform.python_version(),
            "ai_edge_litert": LITERT_VERSION,
            "numpy": np.__version__,
        },
        "protocol": {
            "warmup": args.warmup,
            "min_seconds": args.min_seconds,
            "min_iters": min_iters,
            "batch": 1,
            "timed_region": "interpreter.invoke() only (next image copied into the input tensor before timing)",
            "images": img_info,
            "numerical_check_input": f"{img_info.get('first', 'random')} (first image)",
        },
        "delegation": delegation,
        "runs": [],
    }
    print(f"Model {res['model']['path']}  in {res['model']['input_shape']} out {res['model']['output_shape']}")
    print(f"XNNPACK: {delegation['ops_delegated']}/{delegation['ops_total']} ops, {delegation['partitions']} partitions,"
          f" not delegated {delegation['non_delegated_ops']}")

    if not args.no_check and args.pt.exists():
        res["numerical_check"] = torch_reference_check(args.pt, x, y_tflite)
        print("Numerical check vs PyTorch:", json.dumps(res["numerical_check"]))

    modes = ["xnnpack"] + ([] if args.no_reference else ["builtin_no_xnnpack"])
    samples = {}
    for threads in args.threads:
        for mode in modes:
            it = make_interpreter(model, threads, mode)
            ms = benchmark(it, imgs, args.warmup, args.min_seconds, min_iters)
            del it
            s = stats(ms)
            res["runs"].append({"mode": mode, "threads": threads, "latency": s})
            samples[f"{mode}_t{threads}"] = [round(v, 4) for v in ms.tolist()]
            print(f"{mode:>18} t={threads:<2} median {s['median_ms']:8.2f} ms  P95 {s['p95_ms']:8.2f} ms  "
                  f"mean {s['mean_ms']:8.2f} ms  FPS {s['fps_inverse_mean_latency']:6.1f}  n={s['samples']}",
                  flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (args.out / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    write_markdown(args.out / "metrics.md", res)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
