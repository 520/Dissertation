"""Notebook helpers for reproducible Nsight Systems capture and report display."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'PTQ/profiling_results'
SCRIPT = ROOT / 'PTQ/profile_tensorrt.py'


def find_nsys():
    if path := shutil.which('nsys'):
        return Path(path)
    for base in [Path(os.environ.get('LOCALAPPDATA','')) / 'NVIDIA/NsightSystemsPortable',
                 Path('C:/Program Files/NVIDIA Corporation'),
                 Path('C:/Program Files/NVIDIA GPU Computing Toolkit')]:
        if base.exists():
            paths = [path for path in base.rglob('nsys.exe') if 'arm' not in str(path).lower()]
            if paths:
                return sorted(paths)[-1]
    raise FileNotFoundError('Nsight Systems nsys.exe not found. Install NVIDIA Nsight Systems or add it to PATH.')


def execute(command, log):
    OUT.mkdir(exist_ok=True)
    print(subprocess.list2cmdline([str(value) for value in command]))
    with (OUT/log).open('w',encoding='utf-8') as handle:
        process = subprocess.run([str(value) for value in command],cwd=ROOT,
                                 stdout=handle,stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError((OUT/log).read_text(encoding='utf-8',errors='replace')[-8000:])
    print(f'Saved log: {OUT/log}')


def measure(key, mode, iterations=100):
    execute([sys.executable,SCRIPT,'--model',key,'--mode',mode,'--iterations',iterations],f'{key}_{mode}.log')


def capture(key, iterations=100):
    # Precapture projection retains TRT layer NVTX annotations from the graph
    # constructed before cudaProfilerStart. Only steady-state inference is captured.
    execute([find_nsys(),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
             '--cuda-graph-trace=node:nvtx-precapture','--capture-range=cudaProfilerApi',
             '--capture-range-end=stop','--force-overwrite=true',f'--output={OUT/key}_timeline',
             sys.executable,SCRIPT,'--model',key,'--mode','timeline','--iterations',iterations],
            f'{key}_nsys_capture.log')
    return OUT / f'{key}_timeline.nsys-rep'


def export_stats(key):
    report = OUT/f'{key}_timeline.nsys-rep'
    execute([find_nsys(),'export','--type=sqlite','--force-overwrite=true',
             f'--output={OUT/key}_timeline.sqlite',report],f'{key}_nsys_export.log')
    for name in ['nvtx_sum','nvtx_gpu_proj_sum','nvtx_kern_sum','cuda_gpu_kern_sum','cuda_api_sum','cuda_gpu_mem_time_sum']:
        execute([find_nsys(),'stats','--report',name,'--format=csv','--force-export=true',
                 '--force-overwrite=true','--output',OUT/f'{key}_stats',report],f'{key}_{name}.log')


def show_summary():
    from html import escape
    from IPython.display import HTML, display
    def table(rows):
        keys = list(rows[0])
        content = '<table><thead><tr>' + ''.join(f'<th>{escape(k)}</th>' for k in keys) + '</tr></thead><tbody>'
        for row in rows:
            content += '<tr>' + ''.join(f'<td>{escape(str(row[k]))}</td>' for k in keys) + '</tr>'
        display(HTML(content+'</tbody></table>'))
    stage_rows=[]
    for key in ['kitti','voc']:
        data = json.loads((OUT/f'{key}_stages.json').read_text(encoding='utf-8'))
        for name, value in data['stages'].items():
            stage_rows.append({'Model':key,'Stage':name,'Mean (ms)':round(value['mean_ms'],4),
                               'Median (ms)':round(value['median_ms'],4),'P95 (ms)':round(value['p95_ms'],4),
                               'Mean share (%)':round(value['mean_ms']/data['e2e']['mean_ms']*100,1)})
    table(stage_rows)
    for key in ['kitti','voc']:
        data=json.loads((OUT/f'{key}_layers.json').read_text(encoding='utf-8'))
        print(key, data['method'], '| profiler overhead included')
        table([{'Rank':i+1,'TensorRT fused layer':r['layer'],'Mean (ms)':round(r['mean_ms'],4),
                'P95 (ms)':round(r['p95_ms'],4),'Layer time share (%)':round(r['share_pct'],2)}
               for i,r in enumerate(data['layers'][:10])])
    return stage_rows


def plot_timelines():
    """Render actual Nsight SQLite timestamps (CPU NVTX / CUDA kernels / copies)."""
    import sqlite3
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = plt.get_cmap('tab10').colors
    labels = ['CPU letterbox','CPU tensor view','H2D uint8','GPU preprocess enqueue',
              'TRT graph enqueue + D2D','NMS','Boxes / Results','Final GPU wait']
    for key in ['kitti','voc']:
        with sqlite3.connect(OUT/f'{key}_timeline.sqlite') as db:
            ranges = list(db.execute('''SELECT n.start,n.end,COALESCE(n.text,s.value)
                FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
                WHERE n.end IS NOT NULL'''))
            frames = [r for r in ranges if r[2] and r[2].startswith(f'E2E/{key}/')][5:]
            median = np.median([r[1]-r[0] for r in frames])
            first,last,name = min(frames,key=lambda r:abs(r[1]-r[0]-median))
            stages = [r for r in ranges if r[0]>=first and r[1]<=last and r[2] and
                      (r[2][:2].isdigit() or r[2]=='Framework_GPU_sync')]
            kernels = list(db.execute('''SELECT start,end,graphId FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE start>=? AND end<=?''',(first,last)))
            copies = list(db.execute('''SELECT start,end,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE start>=? AND end<=?''',(first,last)))
        fig,ax=plt.subplots(figsize=(12,4.2))
        for begin,end,stage in stages:
            index=int(stage[:2])-1 if stage[:2].isdigit() else None
            ax.broken_barh([((begin-first)/1e6,(end-begin)/1e6)],(2.65,.5),
                          facecolors=colors[index] if index is not None else '#bdbdbd')
            if (end-begin)/1e6 > .20:
                ax.text(((begin+end)/2-first)/1e6,2.90,stage[:2] if index is not None else 'GPU wait',
                        ha='center',va='center',fontsize=8)
        for begin,end,graph in kernels:
            ax.broken_barh([((begin-first)/1e6,(end-begin)/1e6)],(1.65,.5),
                          facecolors='#225ea8' if graph else '#f16913')
        for begin,end,kind in copies:
            ax.broken_barh([((begin-first)/1e6,(end-begin)/1e6)],(.65,.5),facecolors='#756bb1')
        ax.set(yticks=[.9,1.9,2.9],yticklabels=['CUDA memcpy','CUDA kernels','CPU NVTX ranges'],
               xlabel='Time from E2E range start (ms)',xlim=(0,(last-first)/1e6),ylim=(.3,3.6),
               title=f'{key.upper()} | {name.split("/")[-1]} | Nsight Systems measured timeline')
        ax.grid(axis='x',alpha=.2)
        handles=[Patch(color=colors[i],label=f'{i+1:02d} {label}') for i,label in enumerate(labels)]
        handles += [Patch(color='#225ea8',label='TRT graph kernels'),Patch(color='#f16913',label='Other kernels'),
                    Patch(color='#bdbdbd',label='Framework GPU synchronization')]
        fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.53,-.02))
        fig.tight_layout(rect=(0,.25,1,1))
        for suffix in ['png','svg','pdf']:
            fig.savefig(OUT/f'{key}_timeline.{suffix}',dpi=180,bbox_inches='tight')
        plt.show()
        plt.close(fig)


def write_report():
    import csv
    lines=['# TensorRT profiling: KITTI / VOC YOLOv8n FP32','',
           'Batch=1, imgsz=640, conf=0.25, IoU=0.70; 30 warmup predictions, 100 measured images.',
           'Preloaded BGR input → CPU letterbox → H2D uint8 → GPU layout/cast/normalization → '
           'D2D + TensorRT CUDA Graph → NMS → box scaling/Results → final synchronization.',
           'Disk I/O, decoding, initialization and exporting detections to CPU are excluded.', '',
           '## Synchronized diagnostic decomposition', '',
           'Stage synchronization changes scheduling. Means are additive; individual stage P95 values are not.', '',
           '| Stage | KITTI mean (ms) | VOC mean (ms) |','|---|---:|---:|']
    values={key:json.loads((OUT/f'{key}_stages.json').read_text(encoding='utf-8')) for key in ['kitti','voc']}
    for stage in values['kitti']['stages']:
        lines.append(f"| {stage} | {values['kitti']['stages'][stage]['mean_ms']:.4f} | {values['voc']['stages'][stage]['mean_ms']:.4f} |")
    lines += ['', '## TensorRT layer hotspots', '',
              'Original engines, separate execution context with CUDA Graph captured with IProfiler enabled. '
              '100 replays of one fixed representative input. Layer CUDA events add significant overhead; '
              'these values rank fused layers and cannot replace uninstrumented baseline latency.', '']
    for key in ['kitti','voc']:
        layer=json.loads((OUT/f'{key}_layers.json').read_text(encoding='utf-8'))
        lines += [f'### {key.upper()}: {len(layer["layers"])} fused layers', '',
                  '| Rank | Layer | Mean (ms) | P95 (ms) | Share (%) |','|---:|---|---:|---:|---:|']
        for i,row in enumerate(layer['layers'][:10]):
            escaped=row['layer'].replace('|','\\|')
            lines.append(f"| {i+1} | {escaped} | {row['mean_ms']:.4f} | {row['p95_ms']:.4f} | {row['share_pct']:.2f} |")
        lines += ['']
    lines += ['## Nsight Systems results', '',
              'Captured actual deployment CUDA Graph path, node-level CUDA trace + NVTX, cudaProfilerApi capture '
              'after warmup. CPU stage NVTX durations are launch/host durations, not GPU execution durations. '
              'Gray Framework_GPU_sync ranges annotate the existing Ultralytics Profile boundary waits. '
              'GPU projected ranges measure first-to-last GPU operation span, including gaps; '
              'they can overlap and must not be added. Kernel summary groups identical kernel names across layers.', '']
    for key in ['kitti','voc']:
        trace=json.loads((OUT/f'{key}_timeline.json').read_text(encoding='utf-8'))
        lines += [f'### {key.upper()}', '',
                  f"Trace E2E mean {trace['e2e']['mean_ms']:.3f} ms; median {trace['e2e']['median_ms']:.3f} ms; P95 {trace['e2e']['p95_ms']:.3f} ms (Nsight overhead included).",'',
                  '| GPU projected range | Mean span (ms) | Median span (ms) |', '|---|---:|---:|']
        with (OUT/f'{key}_stats_nvtx_gpu_proj_sum.csv').open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                if row['Range'].startswith(':0'):
                    lines.append(f"| {row['Range']} | {float(row['Proj Avg (ns)'])/1e6:.4f} | {float(row['Proj Med (ns)'])/1e6:.4f} |")
        lines += ['',f'![{key} timeline]({key}_timeline.png)','']
    lines += ['## Open in Nsight Systems', '',
              'Open `kitti_timeline.nsys-rep` and `voc_timeline.nsys-rep` in Nsight Systems. '
              'Expand CUDA HW / Streams and the CPU thread NVTX rows. Search `E2E/`, `05_TensorRT`, '
              'and `06_NMS`; select a steady-state image and zoom to the range. '
              'Follow CUDA API correlation to graph nodes and inspect kernel/copy durations.', '',
              'Engines were built with LAYER_NAMES_ONLY. Layer ranking works, but detailed tensor shapes/tactics '
              'are not embedded; enabling DETAILED requires building a new engine.', '',
              'Official references: [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), '
              '[TensorRT IExecutionContext](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/ExecutionContext.html).']
    (OUT/'profiling_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return OUT/'profiling_report.md'
