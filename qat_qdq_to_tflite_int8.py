"""Convert a ModelOpt QAT Q/DQ ONNX (TensorRT-style) into a real full-INT8 TFLite model for XNNPACK.

A Q/DQ ONNX cannot be turned into an integer TFLite directly (onnx2tf would only emulate the fake-quant in FP32), so:

1. Weights: every QuantizeLinear -> DequantizeLinear pair on a weight initializer is folded into the dequantised
   (fake-quantised) FP32 weight, i.e. the exact INT8 grid learned by QAT. TFLite's symmetric per-channel weight
   quantisation (scale = max|w| / 127) then lands on (almost) the same grid.
2. Activations: Q/DQ pairs are removed and TFLite re-calibrates activation ranges on the same calibration set QAT used
   (KITTI train, 512 images picked with torch.linspace over the sorted file list, letterbox 640, as
   QAT/modelopt_qat.select_calibration_images).
3. The resulting FP32 ONNX (QAT weights) goes through Ultralytics' own onnx2saved_model(quantize=8) path
   (onnx2tf, per-channel INT8), which writes *_integer_quant.tflite (INT8 kernels, FP32 input/output) and
   *_full_integer_quant.tflite (INT8 input/output) among others.

Usage:
    .venv/Scripts/python.exe qat_qdq_to_tflite_int8.py --onnx <qat .onnx>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent
DEFAULT_CALIB = ROOT / "PTQ/datasets/kitti/images/train"
DEFAULT_VAL = ROOT / "PTQ/datasets/kitti/images/val"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def const_values(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    vals = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for n in graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    vals[n.output[0]] = numpy_helper.to_array(a.t)
    return vals


def strip_qdq(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict]:
    """Fold weight Q/DQ into fake-quantised FP32 initializers and drop activation Q/DQ."""
    g = model.graph
    vals = const_values(g)
    inits = {i.name for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    rename: dict[str, str] = {}
    new_inits, drop = [], set()
    stats = collections.Counter()
    for dq in g.node:
        if dq.op_type != "DequantizeLinear":
            continue
        q = producer.get(dq.input[0])
        if q is None or q.op_type != "QuantizeLinear":
            raise SystemExit(f"DequantizeLinear {dq.name} not fed by QuantizeLinear")
        x = q.input[0]
        scale = vals[q.input[1]].astype(np.float32)
        zp = vals[q.input[2]] if len(q.input) > 2 and q.input[2] else np.zeros((), np.uint8)
        axis = next((a.i for a in q.attribute if a.name == "axis"), 1)
        if x in inits:  # weight: fold to the dequantised INT8 grid (ONNX QuantizeLinear semantics)
            w = vals[x].astype(np.float32)
            shape = [1] * w.ndim
            if scale.ndim == 1 and scale.size > 1:
                shape[axis] = -1
            s = scale.reshape(shape)
            z = zp.astype(np.int32).reshape(shape) if zp.ndim else zp.astype(np.int32)
            info = np.iinfo(zp.dtype)
            wq = np.clip(np.rint(w / s) + z, info.min, info.max)
            wfq = ((wq - z) * s).astype(np.float32)
            name = f"{x}__qat_int8_grid"
            new_inits.append(numpy_helper.from_array(wfq, name))
            rename[dq.output[0]] = name
            stats["weight_qdq_folded"] += 1
            stats["weight_per_channel" if scale.size > 1 else "weight_per_tensor"] += 1
        else:  # activation: bypass, TFLite re-calibrates
            rename[dq.output[0]] = x
            stats["activation_qdq_removed"] += 1
        drop.update({q.output[0], dq.output[0]})  # nodes identified by their (unique) output tensor

    def resolve(t: str) -> str:
        while t in rename:
            t = rename[t]
        return t

    keep = [n for n in g.node if n.output[0] not in drop]
    for n in keep:
        for i, t in enumerate(n.input):
            n.input[i] = resolve(t)
    for o in g.output:
        o.name = resolve(o.name)
    g.initializer.extend(new_inits)
    del g.node[:]
    g.node.extend(keep)

    # prune dead nodes / initializers (original FP32 weights, Q/DQ scale constants)
    used = {o.name for o in g.output}
    alive = []
    for n in reversed(list(g.node)):
        if any(o in used for o in n.output):
            alive.append(n)
            used.update(n.input)
    alive.reverse()
    removed_nodes = len(g.node) - len(alive)
    del g.node[:]
    g.node.extend(alive)
    dead = [i for i in g.initializer if i.name not in used]
    for i in dead:
        g.initializer.remove(i)
    stats["dead_nodes_removed"] = removed_nodes
    stats["dead_initializers_removed"] = len(dead)
    onnx.checker.check_model(model)
    return model, dict(stats)


def normalize_box_head(model: onnx.ModelProto, imgsz: int) -> tuple[onnx.ModelProto, dict]:
    """Make the Detect head emit xywh normalised to [0, 1] (as Ultralytics' TF/EdgeTPU exports do).

    The exported head computes boxes in grid units (anchors up to ~80) and multiplies by the stride at the end, then
    concatenates pixel boxes (0..640) with class scores (0..1). With full-integer quantisation that concat shares one
    INT8 scale, which wipes out the scores. Here DFL distances and anchors are multiplied by stride / imgsz up front
    and the final stride multiplication is dropped, so every head tensor stays in [0, 1].
    Output boxes must be multiplied by imgsz to get pixels.
    """
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    cat = producer[g.output[0].name]
    assert cat.op_type == "Concat", cat.op_type
    smul = producer[cat.input[0]]
    assert smul.op_type == "Mul", smul.op_type
    stride_name = next(t for t in smul.input if t in inits)
    strides = numpy_helper.to_array(inits[stride_name]).astype(np.float32).reshape(-1)  # (A,)
    n_anchor = strides.size
    norm = (strides / imgsz).reshape(1, 1, n_anchor).astype(np.float32)

    # anchor constants (1, 2, A) used by the Sub/Add of dist2bbox, and the Slices that feed them
    anchors = [n for n, i in inits.items() if tuple(i.dims) == (1, 2, n_anchor)]
    assert len(anchors) in (1, 2), anchors
    dist = set()
    for n in g.node:
        if n.op_type in ("Sub", "Add") and any(t in anchors for t in n.input):
            other = next(t for t in n.input if t not in anchors)
            sl = producer[other]
            assert sl.op_type == "Slice", sl.op_type
            dist.add(sl.input[0])
    assert len(dist) == 1, dist
    dist = dist.pop()

    for name in anchors:
        arr = numpy_helper.to_array(inits[name]).astype(np.float32) * norm
        inits[name].CopyFrom(numpy_helper.from_array(arr, name))
    norm_name, dist_n = "head_box_norm__stride_div_imgsz", f"{dist}__normalized"
    g.initializer.append(numpy_helper.from_array(norm, norm_name))
    for n in g.node:
        if n.op_type == "Slice" and n.input[0] == dist:
            n.input[0] = dist_n
    mul = onnx.helper.make_node("Mul", [dist, norm_name], [dist_n], name="head_box_normalize")
    nodes = list(g.node)
    pos = nodes.index(producer[dist]) + 1
    nodes.insert(pos, mul)
    cat.input[0] = next(t for t in smul.input if t != stride_name)  # drop "* stride"
    nodes.remove(smul)
    del g.node[:]
    g.node.extend(nodes)
    if not any(stride_name in n.input for n in g.node):
        g.initializer.remove(inits[stride_name])
    onnx.checker.check_model(model)
    return model, {"boxes": "xywh normalised by imgsz", "imgsz": imgsz, "anchors": n_anchor,
                   "anchor_constants_scaled": len(anchors)}


def letterbox(img: np.ndarray, size: int = 640, color: int = 114) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = int(round((size - nh) / 2 - 0.1)), int(round((size - nw) / 2 - 0.1))
    return cv2.copyMakeBorder(img, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT,
                              value=(color, color, color))


def select_images(root: Path, samples: int) -> list[Path]:
    """Same selection rule as QAT/modelopt_qat.select_calibration_images (sorted, torch.linspace indices)."""
    import torch

    paths = sorted({p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES})
    count = min(samples, len(paths))
    return [paths[i] for i in torch.linspace(0, len(paths) - 1, count).long().tolist()]


def load_bhwc(paths: list[Path], size: int) -> np.ndarray:
    """BHWC float32 RGB in [0, 255] (Ultralytics' calibration format for onnx2saved_model)."""
    out = np.empty((len(paths), size, size, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        out[i] = letterbox(cv2.imread(str(p)), size)[..., ::-1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True, help="ModelOpt Q/DQ ONNX")
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL, help="images for the fidelity check")
    ap.add_argument("--check-images", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--skip-convert", action="store_true",
                    help="reuse existing TFLite files in the output dir; only redo the fidelity check and report")
    args = ap.parse_args()

    src = args.onnx.resolve()
    stem = src.stem
    float_onnx = src.with_name(f"{stem}_qatw_fp32.onnx")
    out_dir = src.parent / "tflite_int8"  # short: onnx2tf file names are long and Windows MAX_PATH is 260
    longest = out_dir / f"{float_onnx.stem}_full_integer_quant_with_int16_act.tflite"
    if sys.platform == "win32" and len(str(longest)) >= 260:
        sys.exit(f"Output path would be {len(str(longest))} chars (>= 260, Windows MAX_PATH): {longest}\n"
                 "Move/rename the model to a shorter path first.")

    # 1) strip Q/DQ
    qdq = onnx.load(str(src))
    import onnxruntime as ort

    stats = head = None
    if not args.skip_convert:
        import onnxslim

        stripped, stats = strip_qdq(onnx.load(str(src)))
        stripped = onnxslim.slim(stripped)
        normed, head = normalize_box_head(onnx.ModelProto.FromString(stripped.SerializeToString()), args.imgsz)
        # the head rewrite must be exact: compare against the un-normalised stripped graph
        x = np.random.default_rng(0).random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
        ya = ort.InferenceSession(stripped.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
        yb = ort.InferenceSession(normed.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
        head["rewrite_box_max_abs_diff_px"] = float(np.abs(yb[:, :4] * args.imgsz - ya[:, :4]).max())
        head["rewrite_score_max_abs_diff"] = float(np.abs(yb[:, 4:] - ya[:, 4:]).max())
        print(f"Head normalised: {head}", flush=True)
        onnx.save(normed, str(float_onnx))
        print(f"Stripped Q/DQ: {stats} -> {float_onnx.name}", flush=True)
        print("ops:", dict(collections.Counter(n.op_type for n in normed.graph.node)), flush=True)
    else:
        _, stats = strip_qdq(onnx.load(str(src)))  # recompute stats for the report only

    # 2) fidelity of the stripped graph vs the Q/DQ graph (onnxruntime, CPU); float_onnx boxes are normalised
    val_paths = sorted(p for p in args.val.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)[: args.check_images]
    x_val = load_bhwc(val_paths, args.imgsz) / 255.0
    so = ort.SessionOptions()
    s_qdq = ort.InferenceSession(qdq.SerializeToString(), so, providers=["CPUExecutionProvider"])
    s_fp = ort.InferenceSession(str(float_onnx), so, providers=["CPUExecutionProvider"])
    y_qdq = np.concatenate([s_qdq.run(None, {"images": x[None].transpose(0, 3, 1, 2)})[0] for x in x_val])
    y_fp = np.concatenate([s_fp.run(None, {"images": x[None].transpose(0, 3, 1, 2)})[0] for x in x_val])
    y_fp[:, :4] *= args.imgsz  # normalised -> pixels

    # 3) Ultralytics onnx2saved_model with INT8 calibration
    calib_paths = select_images(args.calib, args.calib_samples)
    if not args.skip_convert:
        from ultralytics.utils.export.tensorflow import onnx2saved_model

        if out_dir.is_dir():
            import shutil

            shutil.rmtree(out_dir)  # stale TFLite variants from an earlier run
        t0 = time.time()
        images = load_bhwc(calib_paths, args.imgsz)
        print(f"Calibration: {len(calib_paths)} images from {args.calib} ({images.nbytes / 1e9:.2f} GB) "
              f"in {time.time() - t0:.1f}s", flush=True)
        t0 = time.time()
        onnx2saved_model(str(float_onnx), out_dir, quantize=8, images=images, prefix="TFLite INT8:")
        del images
        print(f"onnx2saved_model done in {time.time() - t0:.0f}s", flush=True)
    # same clean-up as the Ultralytics exporter: drop int16-activation variants and the calibration dump
    for f in [*out_dir.glob("*_with_int16_act.tflite"), *out_dir.glob("tmp_tflite_int8_calibration_images.npy")]:
        f.unlink()

    # 4) TFLite INT8 vs Q/DQ ONNX on the same check images
    from ai_edge_litert.interpreter import Interpreter

    tfl = sorted(out_dir.glob("*_integer_quant.tflite"))
    tfl = [p for p in tfl if "full_integer" not in p.name][0]
    it = Interpreter(model_path=str(tfl), num_threads=4)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    y_tfl = []
    for x in x_val:
        it.set_tensor(ind["index"], x[None].astype(np.float32))
        it.invoke()
        y_tfl.append(it.get_tensor(outd["index"]).copy())
    y_tfl = np.concatenate(y_tfl)
    y_tfl[:, :4] *= args.imgsz  # normalised -> pixels

    def compare(a: np.ndarray, b: np.ndarray) -> dict:
        sa, sb = a[:, 4:].max(1), b[:, 4:].max(1)  # per-anchor best class score
        overlap = [len(set(np.argsort(-u)[:100]) & set(np.argsort(-v)[:100])) for u, v in zip(sa, sb)]
        det = [(u > 0.25).sum() for u in sa], [(v > 0.25).sum() for v in sb]
        return {"box_mean_abs_diff_px": float(np.abs(a[:, :4] - b[:, :4]).mean()),
                "score_max_abs_diff": float(np.abs(a[:, 4:] - b[:, 4:]).max()),
                "score_mean_abs_diff": float(np.abs(a[:, 4:] - b[:, 4:]).mean()),
                "top100_anchor_overlap_mean": float(np.mean(overlap)),
                "anchors_over_0.25_total": [int(sum(det[0])), int(sum(det[1]))]}

    report = {
        "source_onnx": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
        "strip_stats": stats,
        "float_onnx": float_onnx.name,
        "head_normalisation": head,
        "output_boxes": "xywh normalised to [0, 1]; multiply by imgsz for pixels",
        "calibration": {"images": len(calib_paths), "root": str(args.calib), "rule": "sorted + torch.linspace",
                        "first": calib_paths[0].name, "last": calib_paths[-1].name, "letterbox": args.imgsz},
        "tflite_int8": tfl.name,
        "tflite_input": {"dtype": np.dtype(ind["dtype"]).name, "shape": [int(v) for v in ind["shape"]]},
        "check_images": len(val_paths),
        "stripped_fp32_vs_qdq_onnx": compare(y_fp, y_qdq),
        "tflite_int8_vs_qdq_onnx": compare(y_tfl, y_qdq),
    }
    (out_dir / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
