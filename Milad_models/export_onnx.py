"""Export the pruned YOLOv8 KITTI model to ONNX.

Usage:  .venv\\Scripts\\python.exe export_onnx.py [path\\to\\model.pt]
"""
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

src = Path(sys.argv[1] if len(sys.argv) > 1 else r"yolov8_kitti_pruning\model.pt").resolve()
obj = torch.load(src, map_location="cpu", weights_only=False)
print("loaded object type:", type(obj).__name__)

def fix_pruned_c2f(model):
    """Channel pruning can leave C2f.c stale (e.g. c=64 while cv1 outputs 104 channels).
    forward() uses chunk(2) and still works, but ONNX export switches to forward_split(),
    which splits by (c, c). Reset c to cv1_out // 2, which is what chunk(2) and the bottlenecks use."""
    from ultralytics.nn.modules.block import C2f

    x = torch.zeros(1, 3, 640, 640)
    model.eval()
    with torch.no_grad():
        ref = model(x)[0]
    for name, m in model.named_modules():
        if isinstance(m, C2f):
            out = m.cv1.conv.out_channels
            if m.c * 2 != out:
                assert out % 2 == 0 and all(b.cv1.conv.in_channels == out // 2 for b in m.m), name
                print(f"fix {name}: c {m.c} -> {out // 2}")
                m.c = out // 2
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, C2f):
                m.forward = m.forward_split  # same path the ONNX exporter uses
        diff = (model(x)[0] - ref).abs().max().item()
        for m in model.modules():
            if isinstance(m, C2f):
                del m.forward
    print(f"forward_split vs forward max abs diff: {diff:.3e}")
    return model


if isinstance(obj, torch.nn.Module):
    # Pruned model saved as a bare nn.Module -> wrap it in an Ultralytics-style checkpoint.
    model = fix_pruned_c2f(obj.float())
    ckpt = {"model": model, "train_args": dict(getattr(model, "args", {}) or {})}
    ckpt_path = src.with_name(src.stem + "_ckpt.pt")
    torch.save(ckpt, ckpt_path)
    print("wrapped bare module into", ckpt_path)
else:
    ckpt_path = src

yolo = YOLO(str(ckpt_path))
n_params = sum(p.numel() for p in yolo.model.parameters())
print(f"params: {n_params:,}  nc: {yolo.model.nc if hasattr(yolo.model, 'nc') else '?'}  names: {yolo.names}")

out = yolo.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, batch=1)
out = Path(out)
final = src.with_suffix(".onnx")
if out.resolve() != final:
    out.replace(final)
print("ONNX saved to:", final)
