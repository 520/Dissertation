"""Work around an onnx2tf constant-permutation bug that corrupts Gather look-up tables (e.g. LUTSiLU).

onnx2tf (utils/common_functions.get_constant_or_variable) assumes every 1-D constant with more than two elements is
an NCHW-ordered shape/permutation vector and reorders it NCHW -> NHWC: new[k] = old[perm[k]] with
perm = [0, 2, 3, ..., n-1, 1]. For a LUT fed to Gather this moves table entry 1 to the end, so every lookup returns
the *next* entry and the converted TFLite model is numerically wrong (Activation/lut_silu.LUTSiLU: values[17] and
deltas[16] tables).

prepermute_gather_tables() stores each such table in the inverse order, so onnx2tf's permutation restores it.

A second problem with the same model: after the Where-terminated LUT activation onnx2tf no longer knows the Detect-head
conv outputs are NHWC and reshapes them to [1, C, H*W] without transposing first. head_reshape_replacements() writes an
onnx2tf param-replacement file that inserts the missing NHWC -> NCHW transpose before those Reshapes.

Always verify the converted model against onnxruntime on the original ONNX afterwards (main() does).

Usage (FP32, Ultralytics' onnx2saved_model path, same as `yolo export format=saved_model`):
    .venv/Scripts/python.exe onnx2tf_lut_fix.py --onnx models/activation/8n_float32_lut16seg_minus5_5.onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent


def onnx2tf_1d_perm(n: int) -> list[int]:
    return [0, *range(2, n), 1]


def prepermute_gather_tables(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace each 1-D (size > 2) constant Gather data input by its inverse-permuted copy."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    consts = {n.output[0]: n for n in g.node if n.op_type == "Constant"}
    fixed = 0
    for n in g.node:
        if n.op_type != "Gather":
            continue
        name = n.input[0]
        if name in inits:
            table = numpy_helper.to_array(inits[name])
        elif name in consts:
            attr = next((a for a in consts[name].attribute if a.name == "value"), None)
            if attr is None:
                continue
            table = numpy_helper.to_array(attr.t)
        else:
            continue  # data is a computed tensor, not affected
        if table.ndim != 1 or table.size <= 2:
            continue
        perm = onnx2tf_1d_perm(table.size)
        stored = np.empty_like(table)
        stored[perm] = table  # onnx2tf then computes stored[perm] -> table
        new_name = f"{name}__onnx2tf_prepermuted_{fixed}"
        g.initializer.append(numpy_helper.from_array(stored, new_name))
        n.input[0] = new_name
        fixed += 1
    return model, fixed


def head_reshape_replacements(model: onnx.ModelProto) -> list[dict]:
    """onnx2tf param-replacement entries forcing NHWC -> NCHW before every Conv -> rank-3 Reshape.

    With the LUT activation (ends in Where) onnx2tf loses track of the NHWC layout and reshapes the Detect-head conv
    outputs ([1, H, W, C] in TF) straight to [1, C, H*W], scrambling every anchor. A pre-process transpose
    [0, 3, 1, 2] on the Reshape input restores ONNX semantics.
    """
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(next(a for a in n.attribute if a.name == "value").t)
    ops = []
    for n in g.node:
        if n.op_type != "Reshape" or n.input[0] not in producer or producer[n.input[0]].op_type != "Conv":
            continue
        shape = consts.get(n.input[1])
        if shape is None or shape.size != 3:
            continue
        ops.append({"op_name": n.name, "param_target": "inputs", "param_name": n.input[0],
                    "pre_process_transpose_perm": [0, 3, 1, 2]})
    return ops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True, help="FP32 ONNX exported by Ultralytics")
    ap.add_argument("--images", type=Path, default=ROOT / "PTQ/datasets/kitti/images/val")
    args = ap.parse_args()

    src = args.onnx.resolve()
    fixed_onnx = src.with_name(f"{src.stem}_lutfix.onnx")
    model, n = prepermute_gather_tables(onnx.load(str(src)))
    onnx.checker.check_model(model)
    onnx.save(model, str(fixed_onnx))
    print(f"pre-permuted {n} Gather tables -> {fixed_onnx.name}", flush=True)

    out_dir = src.with_name(f"{src.stem}_saved_model")
    if out_dir.is_dir():
        shutil.rmtree(out_dir)
    replacements = head_reshape_replacements(model)
    rep_file = src.with_name(f"{src.stem}_onnx2tf_param_replacement.json")
    rep_file.write_text(json.dumps({"format_version": 1, "operations": replacements}, indent=2), encoding="utf-8")
    print(f"onnx2tf param replacement: NHWC->NCHW before {len(replacements)} head Reshapes -> {rep_file.name}",
          flush=True)

    import onnx2tf
    from ultralytics.utils.export.tensorflow import onnx2saved_model

    _convert = onnx2tf.convert  # Ultralytics' onnx2saved_model calls onnx2tf.convert(...); add the replacement file
    onnx2tf.convert = lambda *a, **k: _convert(*a, param_replacement_file=str(rep_file), **k)
    try:
        onnx2saved_model(str(fixed_onnx), out_dir, prefix="TFLite (LUT fix):")
    finally:
        onnx2tf.convert = _convert
    # keep the same file names as `yolo export format=saved_model` (the stem of the original ONNX)
    for f in out_dir.glob(f"{fixed_onnx.stem}_*.tflite"):
        f.rename(f.with_name(f.name.replace(fixed_onnx.stem, src.stem)))

    # verify against onnxruntime on the original (unfixed) ONNX
    import cv2
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    sys.path.insert(0, str(ROOT))
    from tflite_xnnpack_benchmark import letterbox

    tfl = out_dir / f"{src.stem}_float32.tflite"
    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    it = Interpreter(model_path=str(tfl), num_threads=4)
    it.allocate_tensors()
    ii, oo = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    box, score = [], []
    for p in sorted(args.images.glob("*.png"))[:5]:
        x = (letterbox(cv2.imread(str(p)), 640)[..., ::-1].astype(np.float32) / 255.0)[None]
        yo = sess.run(None, {sess.get_inputs()[0].name: x.transpose(0, 3, 1, 2).copy()})[0]
        it.set_tensor(ii, x)
        it.invoke()
        yl = it.get_tensor(oo)
        box.append(np.abs(yl[:, :4] - yo[:, :4]).max())
        score.append(np.abs(yl[:, 4:] - yo[:, 4:]).max())
    report = {"onnx": src.name, "tables_prepermuted": n, "head_reshape_transposes": len(replacements),
              "tflite": tfl.name,
              "check_images": len(box), "box_max_abs_diff_px": float(max(box)), "score_max_abs_diff": float(max(score))}
    (out_dir / "lut_fix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
